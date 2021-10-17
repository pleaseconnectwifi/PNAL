class MiniBatch(object):
    def __init__(self):
        self.ids = []
        self.images = []
        self.labels = []

    def append(self, id, image, label):
        self.ids.append(id)
        self.images.append(image)
        self.labels.append(label)

    def get_size(self):
        return len(self.ids)