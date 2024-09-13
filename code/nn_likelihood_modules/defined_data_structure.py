"""
Defined data structures for the pratical use of training
"""

class experiment_data():
    def __init__(self, lr, batch_size, train_losses, valid_losses, train_accuracies, valid_accuracies, test_loss, test_accuracy, nb_epochs):
        self.lr = lr
        self.batch_size = batch_size
        self.train_losses = train_losses
        self.valid_losses = valid_losses
        self.train_accuracies = train_accuracies
        self.valid_accuracies = valid_accuracies
        self.test_loss = test_loss
        self.test_accuracy = test_accuracy
        self.nb_epochs = nb_epochs   