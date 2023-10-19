from classifier import trainer, datahandler

mytrainer = trainer.Trainer(10, datahandler.MINSTDataHandler.get_data(), 5)


mytrainer.tune('./data/weights/full_metal.keras', './data/stats/full_metal.json', (0.3,), ('relu', 'tanh'), (32, 64, 128, 256, 512), ('relu', 'tanh'), (2, 3), (2, 3, 4), (32, 64, 128), (1, 2))
