import pathlib

data_dir = pathlib.Path("trainingImages")

roses = list(data_dir.glob('uniform/*'))
PIL.Image.open(str(roses[0]))

