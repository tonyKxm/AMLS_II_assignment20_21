import numpy as np


class MixupImageDataGenerator():
    def __init__(self, dataframe, generator,x_col, y_col, directory, batch_size, target_size, alpha=0.2, color_mode = 'rgb',subset=None):
        """Constructor for mixup image data generator.

        Arguments:
            generator {object} -- An instance of Keras ImageDataGenerator.
            directory {str} -- Image directory.
            batch_size {int} -- Batch size.
            target_size {int,int} -- Image size in pixels.

        Keyword Arguments:
            alpha {float} -- Mixup beta distribution alpha parameter. (default: {0.2})
            subset {str} -- 'training' or 'validation' if validation_split is specified in
            `generator` (ImageDataGenerator).(default: {None})
        """

        self.batch_index = 0
        self.batch_size = batch_size
        self.alpha = alpha

        # First iterator yielding tuples of (x, y)
        self.generator1 = generator.flow_from_dataframe(dataframe,directory,x_col = x_col, y_col = y_col,
                                                        target_size=target_size,
                                                        class_mode="categorical",
                                                        batch_size=batch_size,
                                                        shuffle=True,color_mode = color_mode,
                                                        subset=subset)

        # Second iterator yielding tuples of (x, y)
        self.generator2 = generator.flow_from_dataframe(dataframe,directory,x_col = x_col, y_col = y_col,
                                                        target_size=target_size,
                                                        class_mode="categorical",
                                                        batch_size=batch_size,
                                                        shuffle=True,color_mode = color_mode,
                                                        subset=subset)

        # Number of images across all classes in image directory.
        self.n = self.generator1.samples

    def reset_index(self):
        """Reset the generator indexes array.
        """

        self.generator1._set_index_array()
        self.generator2._set_index_array()

    def on_epoch_end(self):
        self.reset_index()

    def reset(self):
        self.batch_index = 0

    def __len__(self):
        # round up
        return (self.n + self.batch_size - 1) // self.batch_size

    def get_steps_per_epoch(self):
        """Get number of steps per epoch based on batch size and
        number of images.

        Returns:
            int -- steps per epoch.
        """

        return self.n // self.batch_size

    def __next__(self):
        """Get next batch input/output pair.

        Returns:
            tuple -- batch of input/output pair, (inputs, outputs).
        """

        if self.batch_index == 0:
            self.reset_index()

        current_index = (self.batch_index * self.batch_size) % self.n
        if self.n > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0
        
        # Get a pair of inputs and outputs from two iterators.
#         X1, y1 = self.generator1.next()
#         X2, y2 = self.generator2.next()
        
#         rand = np.random.rand()
#         if(rand<0.25):
#             X1, y1 = self.generator1.next()
#             return X1, y1
#         elif (rand<0.5 and rand>=0.25):
#             X2, y2 = self.generator2.next()
#             return X2, y2
#         else:
        X1, y1 = self.generator1.next()
        X2, y2 = self.generator2.next()
        # random sample the lambda value from beta distribution.
        temp_size = X1.shape[0]
        l = np.random.beta(self.alpha, self.alpha, temp_size)

        X_l = l.reshape(temp_size, 1, 1, 1)
        y_l = l.reshape(temp_size, 1)

        # Perform the mixup.
        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)
        return X, y

    def __iter__(self):
        while True:
            yield next(self)
