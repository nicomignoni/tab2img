from sklearn.datasets import fetch_covtype
from tab2img.converter import Tab2Img

dataset = fetch_covtype()

train = dataset.data
target = dataset.target

model = Tab2Img()
images = model.fit_transform(train, target)



