from classifier import trainer, datahandler

mytrainer = trainer.Trainer(10, datahandler.MINSTDataHandler.get_data(), 5)


mytrainer.tune('./data/weights/full_metal.keras', './data/stats/full_metal.json', (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), ('relu', 'tanh'), (32, 64, 128, 256, 512), ('relu', 'tanh'), (2, 3 ,4 ,5), (2, 3, 4), (16, 32, 64, 128), (1, 2, 3))
