#%%
from mesh_transform_2d import *
from math import cos, sin, pi
def create_test_im_2d():
    
    im_test = torch.zeros((3,200,300),dtype = torch.float32)
    for i in range(10):
        for j in range(15):
            im_test[0,i*20+1:i*20+19,j*20+1:j*20+19] = cos(pi*i/30)*sin(pi*j/30)
            im_test[1,i*20+1:i*20+19,j*20+1:j*20+19] = cos(pi*i/30)*cos(pi*j/30)
            im_test[2,i*20+1:i*20+19,j*20+1:j*20+19] = sin(pi*i/30)
    im_test = im_test/im_test.max()
    return im_test
#%%
def test_resize(d = 2):
    m1 = torch.stack(
        torch.meshgrid(*[torch.linspace(0,1,2) for i in range(2)]))
    m2 = 0.25+0.5*m1
    im_test = create_test_im_2d()
    
# %%
grid_init = torch.stack(
    torch.meshgrid(*[torch.linspace(-1, 1, steps=i,dtype=torch.float32) for i in [9,13]], 
                   indexing="ij"),dim=2)
image = create_test_im_2d()

# %%

# %%
#test 1: identity transform
image2 = mesh_transform_2d(image, grid_init, grid_init)
plt.imshow(image2.permute([1,2,0]))
plt.show()

# %%
#test 2: simple scaling
grid_target = 2*grid_init
image3 = mesh_transform_2d(image, grid_init, grid_target)
plt.imshow(image3.permute([1,2,0]))
plt.show()



# %%
#test 3 : simple translation
grid_target = 1.2*(grid_init + (torch.rand(grid_init.shape)-0.5)/20)
image4 = mesh_transform_2d(image, grid_init, grid_target)
plt.imshow(image4.permute([1,2,0]))
plt.show()
# %%
#test 3: random shift
grid_target = grid_init + (torch.rand(grid_init.shape)-0.5)/10

image5 = mesh_transform_2d(image3, grid_init, grid_target)
plt.imshow(image5.permute([1,2,0]))
plt.show() 

# %%
#test 3': reversed random shift
image6 = mesh_transform_2d(image5, grid_target,grid_init)

plt.imshow(image6.permute([1,2,0]))
plt.show()
plt.imshow((-image+image6).permute([1,2,0]))
plt.show()

plt.show()
# %%
# %%
def test_visualization():
    grid_init = torch.stack(
        torch.meshgrid(*[torch.linspace(-1, 1, steps=i) for i in [15,10]], 
                       indexing="ij"),dim=2)
    fig,axs = plt.subplots(4,2)
    for ax in axs:
        ax.imshow(image)
    fig.show()


    
# %%
