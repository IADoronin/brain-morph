#%%
import numpy as np
import torch

import cv2 as cv
import matplotlib.pyplot as plt
import warnings
import typing
import torch.nn.functional as F
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
    
    def visualize(self,channel=None):
        """Визуализация 3D тензора максимальными проекциями по осям."""
        
        channels, depth, height, width = self.shape
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        if channel is not None:
            vol = self[channel, :, :].squeeze(0)

            axes[0][0].imshow(torch.max(vol, dim=2)[0].T, cmap='gray')
            axes[0][1].imshow(torch.max(vol, dim=0)[0], cmap='gray')
            axes[1][0].imshow(torch.max(vol, dim=1)[0].T, cmap='gray')

            for ax in (axes[0][0], axes[0][1], axes[1][0]):
                ax.axis('off')
        else:
            if channels >3:
                warnings.warn("Number of channels > 3. Visualizing only first 3 channels as RGB image. The rest will be ignored for visualization.")
            
            axes[0][0].imshow(torch.max(self, dim=3)[0].permute([2,1,0])[:,:,:3])
            axes[0][1].imshow(torch.max(self, dim=1)[0].permute([1,2,0])[:,:,:3])
            axes[1][0].imshow(torch.max(self, dim=2)[0].T)

            for ax in (axes[0][0], axes[0][1], axes[1][0]):
                ax.axis('off')
        
        axes[1][1].axis('off')
        plt.show()
    def normalize(self):
        """
        Нормализует объем так, чтобы значения были в диапазоне [0, 1].
        """
        min_val = torch.min(self)
        max_val = torch.max(self)
        normalized_data = (self - min_val) / (max_val - min_val)
        return Volume(normalized_data, affine=self.affine)
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
test_volume()
# # # %%

# %%
