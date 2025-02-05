img1 = inputs[0][0][0]
img2 = (img1 - img1.min())/(img1.max() - img1.min())*255
img3 = Image.fromarray(img2.permute(1,2,0).cpu().numpy().astype(np.uint8))
from PIL import Image