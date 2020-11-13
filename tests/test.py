from sklearn.datasets import load_digits
from tab2img import Tab2Img

dataset = load_digits()
train = dataset.data
target = dataset.target

model = Tab2Img()
img = model.fit_transform(train ,target)
