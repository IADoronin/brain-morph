#%%
import volume
from mesh_transform_3d import *
from math import cos, sin, pi
import random
import volume
#%%
def gen_rand_color(i=255):
    tetta= (random.random())*pi/3
    phi = random.random()*pi/2

    return torch.tensor([255*sin(tetta)*cos(phi),255*sin(tetta)*sin(phi),255*cos(tetta)],dtype=torch.uint8)


#
#%%
def create_test_im_3d():
    
    im_test = torch.zeros((3,200,300,400),dtype = torch.uint8)
    for i in range(10):
        for j in range(15):
            for k in range(20):
                im_test[:,i*20+1:i*20+19,j*20+1:j*20+19,k*20+1:k*20+19] = gen_rand_color().reshape([3,1,1,1])*torch.ones([3,18,18,18])
    im_test = im_test/im_test.max()
    return im_test

    
# # %%
# # Создаём сетку
grid_init = torch.stack(
    torch.meshgrid(*[torch.linspace(-1, 1, steps=i) for i in [2,3,4]], 
                   indexing="ij"),dim=-1)
print(grid_init.shape)
image = create_test_im_3d()
image = volume.Volume(image).resample([100,150,200])
volume.Volume(image).visualize()

# %%
#test 1: identity transform
image2 = mesh_transform_3d(image, grid_init, grid_init)
volume.Volume(image2).visualize()
volume.Volume(image2-image).visualize()

#%%
#test 2: simple scaling
grid_target = 2*grid_init
image3 = mesh_transform_3d(image, grid_init, grid_target)
volume.Volume(image3).visualize()

#%%
#test 2.1: simple scaling
grid_target = 0.5*grid_init
image3_1 = mesh_transform_3d(image3, grid_init, grid_target)
volume.Volume(image3_1).visualize()
# %%
#test 3 : simple translation
grid_target = grid_init + torch.tensor((0.5,0.5,0.5))
image4 = mesh_transform_3d(image, grid_init, grid_target)
image4.visualize()
# %%
#test 3: random shift
grid_target = grid_init + torch.rand(grid_init.shape)/5

image5 = mesh_transform_3d(image3, grid_init, grid_target)
image5.visualize()
#%%

image5.rotate(45,45,(50,75,100
                     )).visualize()
image3.rotate(45,45,(50,75,100
                     )).visualize()
#%%

#%%
#test 3: random shift
image6 = mesh_transform_3d(image5, grid_target,grid_init)
image6.visualize()



# %%
