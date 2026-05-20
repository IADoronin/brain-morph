# Copyright (C) 2026 Ivan Doronin <iadoronin@yandex.ru>
# This file is part of brain-morph, licensed under GNU GPL v3.0.
# See LICENSE file in the project root for full license text.

#%%
import numpy as np
import torch

import cv2 as cv
import matplotlib.pyplot as plt
import warnings
import typing
import math
import os
import torch.nn.functional as F
try:
    import nibabel as nib
except Exception:
    nib = None


def _free_memory_gb() -> float:
    """Available system RAM in GB, or inf if cannot be determined."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 1e9
    except (AttributeError, ValueError):
        return float("inf")


#Что сделать: добавить разрешение в мкм по разным осям, что актуально для конфокальной микроскопии
class Volume(torch.Tensor):
    @staticmethod
    def __new__(cls, data, affine=None, *args, **kwargs):
        # Создаем экземпляр тензора через as_subclass
        # Это позволяет превратить любой тензор в объект вашего класса без лишнего копирования памяти
        obj = torch.as_tensor(data, *args, **kwargs).as_subclass(cls)
        return obj

    def __init__(self, data, affine=None, *args, **kwargs):
        # Инициализируем дополнительные атрибуты
        self.affine = affine if affine is not None else torch.eye(4)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Этот метод перехватывает все функции PyTorch (add, sub, mul, sum и т.д.)
        """
        if kwargs is None:
            kwargs = {}
        
        # Выполняем стандартную операцию PyTorch
        res = super().__torch_function__(func, types, args, kwargs)
        
        # Если результат — тензор, оборачиваем его обратно в наш класс Volume
        if isinstance(res, torch.Tensor) and not isinstance(res, cls):
            # Пытаемся вытащить affine из аргументов (если он там был)
            # Это позволяет сохранять метаданные при операциях типа vol1 + vol2
            affine = getattr(args[0], 'affine', None) if args else None
            return cls(res, affine=affine)
        
        return res
    
    def visualize(self, channel=None):
        """Визуализация 3D тензора максимальными проекциями по осям."""
        plt.rcParams["image.resample"] = False
        channels, depth, height, width = self.shape

        # Subplot sizes proportional to (D, H, W) so 1 voxel = same physical
        # size in all three projections:
        #   top-left  (H rows × D cols): width_ratio=D, height_ratio=H
        #   top-right (H rows × W cols): width_ratio=W, height_ratio=H
        #   bot-left  (W rows × D cols): width_ratio=D, height_ratio=W
        scale = 10.0 / max(depth + width, height + width)
        fig = plt.figure(figsize=((depth + width) * scale, (height + width) * scale))
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[depth, width],
            height_ratios=[height, width],
            left=0, right=1, top=1, bottom=0, wspace=0, hspace=0,
        )
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])

        if channel is not None:
            vol = self[channel]  # (D, H, W)
            ax00.imshow(torch.max(vol, dim=2)[0].T, cmap='gray', aspect='auto')
            ax01.imshow(torch.max(vol, dim=0)[0],   cmap='gray', aspect='auto')
            ax10.imshow(torch.max(vol, dim=1)[0].T, cmap='gray', aspect='auto')
        else:
            if channels > 3:
                warnings.warn("Number of channels > 3. Visualizing only first 3 channels as RGB image. The rest will be ignored for visualization.")
            ax00.imshow(torch.max(self, dim=3)[0].permute([2, 1, 0])[:, :, :3], aspect='auto')
            ax01.imshow(torch.max(self, dim=1)[0].permute([1, 2, 0])[:, :, :3], aspect='auto')
            ax10.imshow(torch.max(self, dim=2)[0].T,                             aspect='auto')

        for ax in (ax00, ax01, ax10, ax11):
            ax.axis('off')

        plt.show()
    def normalize(self):
        """
        Нормализует объем так, чтобы значения были в диапазоне [0, 1].
        """
        min_val = torch.min(self)
        max_val = torch.max(self)
        normalized_data = (self - min_val) / (max_val - min_val)
        return Volume(normalized_data, affine=self.affine)
    
    def save_nii(self, file_path: str) -> None:
        """Save volume to a NIfTI file.

        Accepts shapes ``(C, D, H, W)`` or ``(D, H, W)``.  Single-channel
        volumes are saved as 3-D ``(D, H, W)``; multi-channel as 4-D
        ``(D, H, W, C)`` following the NIfTI convention.
        """
        if nib is None:
            raise ImportError("nibabel is required. Install with `pip install nibabel`.")
        data = self.detach().cpu().float().numpy()
        if data.ndim == 4:
            # (C, D, H, W) → (D, H, W, C)
            data = data.transpose(1, 2, 3, 0)
            if data.shape[-1] == 1:
                data = data[..., 0]   # drop trivial channel dim
        affine = self.affine.numpy() if isinstance(self.affine, torch.Tensor) else self.affine
        nib.save(nib.Nifti1Image(data, affine), file_path)

    @classmethod
    def load_tiff_series(
        cls,
        file_names_mask: str,
        scale: int = 1,
        ratio: float = 1.0,
        max_gb: float | None = None,
    ) -> 'Volume':
        """Load a 3D volume from a series of per-slice TIFF files.

        Mirrors MATLAB ``mload(FNAMEFMT, DECREASE, RATIO)``.

        Files are matched by glob and sorted by the integer embedded in each
        filename (negative numbers supported, e.g. ``-3``, ``0``, ``2``).
        Each file becomes one input slice along D.

        Args:
            file_names_mask: Glob pattern, e.g. ``'data/s_C001Z*.tif'``.
            scale: XY downsampling factor (DECREASE in MATLAB).
                scale=4 → H//4, W//4 per slice.
            ratio: XY-to-Z resolution ratio (RATIO in MATLAB).
                ``z_step = scale / ratio`` slices are skipped between output
                slices.  ratio=1 (default) → z_step=scale (isotropic after
                downsampling).  When z_step is non-integer, adjacent slices
                are linearly blended, matching the MATLAB behaviour.
            max_gb: RAM budget in GB for the memory warning
                (None = 70 % of free RAM).  No automatic downsampling —
                a warning with a recommended scale is printed instead.

        Returns:
            ``Volume (C, D_out, H//scale, W//scale)`` where
            ``D_out = ceil(n_files / z_step)``.
        """
        import glob
        import re

        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")
        if ratio <= 0:
            raise ValueError(f"ratio must be > 0, got {ratio}")

        paths = glob.glob(file_names_mask)
        if not paths:
            raise FileNotFoundError(f"No files matched: {file_names_mask}")

        def _num(p: str) -> int:
            stem = os.path.splitext(os.path.basename(p))[0]
            nums = re.findall(r'-?\d+', stem)
            if not nums:
                raise ValueError(f"No integer found in filename: {p}")
            return int(nums[-1])

        paths = sorted(paths, key=_num)
        n_files = len(paths)

        # Probe first slice: shape, channel count
        probe = cv.imread(paths[0], cv.IMREAD_UNCHANGED)
        if probe is None:
            raise IOError(f"Cannot read: {paths[0]}")
        h, w = probe.shape[:2]
        n_ch = 1 if probe.ndim == 2 else probe.shape[2]

        z_step = scale / ratio
        d_out = math.ceil(n_files / z_step)
        dh, dw = max(1, h // scale), max(1, w // scale)

        needed_gb = d_out * dh * dw * n_ch * 4 / 1e9  # float32

        if max_gb is None:
            free = _free_memory_gb()
            max_gb = free * 0.7 if free != float("inf") else float("inf")

        if scale == 1 and max_gb > 0 and needed_gb > max_gb:
            rec = math.ceil((needed_gb / max_gb) ** 0.5)
            warnings.warn(
                f"Volume requires ~{needed_gb:.2f} GB but only {max_gb:.2f} GB "
                f"are available. Consider loading with scale={rec} "
                f"(~{needed_gb / rec ** 2:.2f} GB): "
                f"Volume.load_tiff_series('{file_names_mask}', scale={rec})",
                ResourceWarning,
                stacklevel=2,
            )

        def _read_slice(idx: float) -> np.ndarray:
            """Read one input slice; linearly blend neighbours for float index."""
            lo = int(math.floor(idx))
            hi = int(math.ceil(idx))
            if lo == hi or hi >= n_files:
                img = cv.imread(paths[min(lo, n_files - 1)], cv.IMREAD_UNCHANGED)
                if img is None:
                    raise IOError(f"Cannot read: {paths[lo]}")
                return img.astype(np.float32)
            img_lo = cv.imread(paths[lo], cv.IMREAD_UNCHANGED)
            img_hi = cv.imread(paths[hi], cv.IMREAD_UNCHANGED)
            if img_lo is None or img_hi is None:
                raise IOError(f"Cannot read slice {lo} or {hi}")
            t = idx - lo  # weight of hi slice
            return img_lo.astype(np.float32) * (1.0 - t) + img_hi.astype(np.float32) * t

        slices: list[np.ndarray] = []
        i = 0.0
        while i < n_files:
            sl = _read_slice(i)
            if scale > 1:
                sl = cv.resize(sl, (dw, dh), interpolation=cv.INTER_AREA)
            slices.append(sl)
            i += z_step

        data = np.stack(slices, axis=0)  # (D, H, W) or (D, H, W, C)
        if data.ndim == 3:
            data = data[np.newaxis]          # → (1, D, H, W)
        else:
            data = np.moveaxis(data, -1, 0)  # → (C, D, H, W)

        return cls(torch.as_tensor(data))
    @classmethod
    def load_nii(cls, file_path: str, scale: int = 1, max_gb: float | None = None):
        """Load a NIfTI (.nii or .nii.gz) file and return a Volume (C, D, H, W).

        Args:
            file_path: Path to .nii or .nii.gz file.
            scale: Integer downsampling stride applied along each spatial axis on
                load (every scale-th voxel is kept). scale=1 loads at full
                resolution. Use scale=2 to load 1/8 of voxels, scale=4 for 1/64.
            max_gb: RAM budget in GB used only for the memory warning.
                None = 70 % of current free RAM. When the estimated size exceeds
                the budget and scale=1, a warning is printed with the recommended
                scale value. No automatic downsampling is applied — you must pass
                scale explicitly.
        """
        if nib is None:
            raise ImportError("nibabel is required to load NIfTI files. Install with `pip install nibabel`.")
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")

        img = nib.load(file_path)
        shape = img.header.get_data_shape()

        needed_gb = float(np.prod(shape)) * 4 / 1e9  # float32 size at full res

        if max_gb is None:
            free = _free_memory_gb()
            max_gb = free * 0.7 if free != float("inf") else float("inf")

        if scale == 1 and max_gb > 0 and needed_gb > max_gb:
            recommended = math.ceil((needed_gb / max_gb) ** (1.0 / 3))
            warnings.warn(
                f"Volume requires ~{needed_gb:.2f} GB but only {max_gb:.2f} GB are available. "
                f"Consider loading with scale={recommended} "
                f"(~{needed_gb / recommended ** 3:.2f} GB): "
                f"Volume.load_nii('{file_path}', scale={recommended})",
                ResourceWarning,
                stacklevel=2,
            )

        if scale > 1:
            raw = np.asarray(
                img.dataobj[::scale, ::scale, ::scale], dtype=np.float32
            )
            slope, inter = img.header.get_slope_inter()
            if slope is not None:
                raw = raw * float(slope) + float(inter or 0.0)
            data = raw
        else:
            data = img.get_fdata(dtype=np.float32)

        affine = getattr(img, 'affine', None)

        if data.ndim == 3:
            data = data[np.newaxis, ...]
        elif data.ndim == 4:
            data = np.moveaxis(data, -1, 0)
        else:
            raise ValueError(f"Unsupported NIfTI data dimensionality: {data.ndim}")

        tensor = torch.as_tensor(data)
        return cls(tensor, affine=affine)
    def resample(self, new_shape:typing.Tuple[int,int,int], mode:str='trilinear'):
        """
        Изменяет размер объема до новой формы с использованием интерполяции.

        Параметры:
        - new_shape: Кортеж (depth, height, width) для новой формы.
        - mode: Метод интерполяции ('nearest', 'linear', 'bilinear', 'trilinear' и т.д.).

        Возвращает:
        - Новый объем с измененным размером.
        """
        # Добавляем размер батча и каналов для использования с torch.nn.functional.interpolate
        data = self.unsqueeze(0)  # Форма: (1, C, D, H, W)
        
        # Используем interpolate для изменения размера
        resized_data = torch.nn.functional.interpolate(
            data,
            size=new_shape,
            mode=mode,
            align_corners=False if mode in ['linear', 'bilinear', 'trilinear'] else None
        )
        
        # Убираем размер батча обратно
        resized_data = resized_data.squeeze(0)  # Форма: (C, D, H, W)
        
        return Volume(resized_data, affine=self.affine)
    @classmethod
    def load_tiff(file_mask):
        pass

    def rotate(self, theta, phi, center, interpolation='linear'):
        """
        Поворот 3D объема на PyTorch.
        
        :param theta: угол поворота вокруг оси X (в градусах)
        :param phi: угол поворота вокруг оси Y (в градусах)
        :param center: кортеж (x, y, z) в воксельных координатах
        :param interpolation: 'nearest', 'linear', 'bilinear', 'bicubic'
        """
        # Проверить может можно убрать первое измерение
        # Приведение к форме (1, 1, D, H, W) для grid_sample
        if self.ndim == 3:
            volume = self.unsqueeze(0).unsqueeze(0)
        elif self.ndim == 4:
            volume = self.unsqueeze(0)
        else: 
            volume = self
        device = self.device
        dtype = self.dtype
        N, C, D, H, W = volume.shape

        # Маппинг интерполяции
        # grid_sample поддерживает только 'nearest' и 'bilinear' для 5D
        interp_modes = {
            'nearest': 'nearest',
            'linear': 'bilinear',
            'bilinear': 'bilinear',
            'bicubic': 'bilinear' # Замена, так как 5D bicubic не реализован в torch
        }
        mode = interp_modes.get(interpolation, 'bilinear')

        # 1. Создаем сетку координат (от -1 до 1)
        # Координаты в grid_sample: x=W, y=H, z=D
        grid_d = torch.linspace(-1, 1, D, device=device)
        grid_h = torch.linspace(-1, 1, H, device=device)
        grid_w = torch.linspace(-1, 1, W, device=device)
        
        # meshgrid создает сетку (D, H, W)
        mesh_z, mesh_y, mesh_x = torch.meshgrid(grid_d, grid_h, grid_w, indexing='ij')
        
        # Объединяем в (D*H*W, 3) и добавляем единицу для афинных преобразований
        grid = torch.stack((mesh_x, mesh_y, mesh_z, torch.ones_like(mesh_x)), dim=-1).reshape(-1, 4)
        grid = grid.to(dtype)

        # 2. Подготовка матриц вращения (в радианах)
        t = np.radians(theta)
        p = np.radians(phi)
        
        # Матрица вокруг X
        Rx = torch.tensor([
            [1, 0, 0, 0],
            [0, np.cos(t), -np.sin(t), 0],
            [0, np.sin(t), np.cos(t), 0],
            [0, 0, 0, 1]
        ], dtype=dtype, device=device)

        # Матрица вокруг Y
        Ry = torch.tensor([
            [np.cos(p), 0, np.sin(p), 0],
            [0, 1, 0, 0],
            [-np.sin(p), 0, np.cos(p), 0],
            [0, 0, 0, 1]
        ], dtype=dtype, device=device)

        R = Ry @ Rx # Комбинированное вращение

        # 3. Вращение вокруг заданной точки (center)
        # Переводим координаты центра в нормализованный массив [-1, 1]
        center_arr = torch.tensor([center[0], center[1], center[2]], dtype=dtype, device=device)
        norm_factors = torch.tensor([2.0 / (W - 1), 2.0 / (H - 1), 2.0 / (D - 1)], dtype=dtype, device=device)
        c = center_arr * norm_factors - 1.0

        # Матрица переноса центра в начало координат и обратно
        T1 = torch.eye(4, dtype=dtype, device=device)
        T2 = torch.eye(4, dtype=dtype, device=device)

        # Присваиваем вектор переноса как срез (без трёх отдельных переменных)
        T1[:3, 3] = -c
        T2[:3, 3] = c

        # Итоговая матрица: T2 @ R @ T1
        # Но для grid_sample нам нужно ОБРАТНОЕ преобразование (из выходной сетки в исходную)
        # Поэтому инвертируем матрицу
        M = T2 @ R @ T1
        M_inv = torch.inverse(M)

        # 4. Применяем трансформацию к сетке
        # Нам нужны только первые 3 строки (x, y, z)
        transformed_grid = grid @ M_inv.T
        transformed_grid = transformed_grid[:, :3].reshape(1, D, H, W, 3)

        # 5. Ресэмплинг
        rotated_volume = F.grid_sample(
            volume, 
            transformed_grid, 
            mode=mode, 
            padding_mode='zeros', 
            align_corners=True
        )

        return rotated_volume.squeeze()

# --- Пример запуска ---
# brain_tensor = torch.randn(64, 64, 64).cuda()
# rotated = rotate_3d_volume_torch(brain_tensor, 30.0, 45.0, (32, 32, 32), 'linear')


        # Здесь должна быть реализация поворота объема

    def mesh_transform(self, mesh_vertices:torch.Tensor, mesh_displacements:torch.Tensor):
        """
        Применяет трансформацию к объему на основе заданных вершин сетки и смещений.

        Параметры:
        - mesh_vertices: Тензор формы (X_length, Y_length, Z_length 3), содержащий координаты вершин сетки.
        - mesh_displacements: Тензор формы (N, 3), содержащий смещения для каждой вершины.

        Возвращает:
        - Трансформированный объем.
        """
        # Здесь должна быть реализация трансформации объема

        raise NotImplementedError("Метод mesh_transform еще не реализован.")

    
# %%
def test_volume():
    # Создаем случайный 3D тензор с 3 каналами
   # data = torch.rand(3, 100, 100, 100)*0.8
    data = torch.zeros(3, 100, 100, 100)
    data[0, 30:70, 20:80, 40:60] =1
    data[1, 30:70, 20:80, 40:60] =0.5
    data[2, 30:70, 20:80, 40:60] =0.2
      # Добавляем яркий куб в центр
    
    # Создаем объект Volume
    vol = Volume(data)
    
    # Проверяем, что объект является экземпляром Volume и torch.Tensor
    assert isinstance(vol, Volume), "vol должен быть экземпляром Volume"
    assert isinstance(vol, torch.Tensor), "vol должен быть экземпляром torch.Tensor"
    
    print("Форма Volume:", vol.shape)
    vol_reshaped = vol.resample((50, 50, 50), mode='trilinear')
    vol_reshaped.visualize()
    # Проверяем, что данные внутри Volume совпадают с исходными данными
    assert torch.allclose(vol, data), "Данные внутри Volume не совпадают с исходными данными"
    vol.rotate(45,45,(0,0,0),0).visualize()
    # Проверяем работу метода visualize (визуально)
    print("Визуализация Volume:","individual channel visualization")
    vol.visualize(channel=0)
    print("Визуализация Volume:","all channels visualization")
    vol.visualize(None)
    vol.rotate(45,45,(50,50,50),0).visualize()
    print("Все тесты пройдены успешно!")


# # # %%
#test_volume()
# # # %%

# %%
