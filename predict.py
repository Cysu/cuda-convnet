import os
import numpy as np
from scipy.misc import imread, imresize
from gpumodel import IGPUModel, ModelStateException
from convnet import ConvNet
from util import pickle, unpickle

class DataProcessor(object):
    def __init__(self, meta_path, crop_border, multiview_test):
        self.batch_meta = unpickle(meta_path)

        self.border_size = crop_border
        self.num_colors = 3
        self.outer_size = int((self.batch_meta['data_mean'].size / self.num_colors) ** 0.5)
        self.inner_size = self.outer_size - self.border_size*2
        self.multiview = multiview_test
        self.num_views = 5 * 2
        self.data_mult = self.num_views if self.multiview else 1

        self.label_types = self.batch_meta['label_types']
        self.label_names = self.batch_meta['label_names']
        
        self.data_mean = self.batch_meta['data_mean'].reshape((self.num_colors, self.outer_size, self.outer_size))
        self.data_mean = self.data_mean[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size]
        self.data_mean = self.data_mean.reshape((self.get_data_dims(), 1))

        self.cropped_data = np.zeros((self.get_data_dims(), self.data_mult), dtype=np.single)

    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.inner_size**2 * 3
        elif self.label_types[idx - 1] == 'multi-class':
            return 1
        elif self.label_types[idx - 1] == 'multi-label':
            return len(self.label_names[idx - 1])
        else:
            raise ValueError("Invalid label_types in batch_meta")

    def prep(self, img):
        if img.ndim != 3:
            img = np.tile(img[:,:,np.newaxis], 3)
        if img.shape != (self.outer_size, self.outer_size, self.num_colors):
            img = imresize(img, [self.outer_size, self.outer_size])
        img = np.rollaxis(img, 2).astype(np.single)
        return img.ravel()

    def process(self, image):
        data = self.prep(image)[:,np.newaxis]
        self._trim_borders(data, self.cropped_data)
        self.cropped_data -= self.data_mean
        fake_labels = [np.zeros((self.get_data_dims(i), self.data_mult), dtype=np.single) for i in xrange(1, len(self.label_types) + 1)]
        return [self.cropped_data] + fake_labels

    def _trim_borders(self, x, target):
        y = x.reshape(self.num_colors, self.outer_size, self.outer_size, x.shape[1])
        if self.multiview:
            start_positions = [(0,0),  (0, self.border_size*2),
                               (self.border_size, self.border_size),
                               (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
            end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
            for i in xrange(self.num_views/2):
                pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
                target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
        else:
            pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
            target[:,:] = pic.reshape((self.get_data_dims(), 1))
       
class ConvNetPredict(ConvNet):
    def __init__(self, model_path, data_processor, gpu, layers):
        op = ConvNetPredict.get_options_parser()
        op.set_value('load_file', model_path)
        op.set_value('gpu', str(gpu))

        load_dic = IGPUModel.load_checkpoint(model_path)
        old_op = load_dic["op"]
        old_op.merge_from(op)
        op = old_op
        op.eval_expr_defaults()

        ConvNet.__init__(self, op, load_dic)

        self.dp = data_processor
        self.ftr_layer_idx = map(self.get_layer_idx, layers)
        
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('gpu', 'load_file'):
                op.delete_option(option)
        return op

    def predict(self, image):
        data = self.dp.process(image)
        num_ftrs = [self.layers[i]['outputs'] for i in self.ftr_layer_idx]
        ftrs = [np.zeros((data[0].shape[1], k) , dtype=np.single) for k in num_ftrs]
        self.libmodel.startFeatureWriter(data, ftrs, self.ftr_layer_idx)
        self.libmodel.finishBatch()
        ftrs = [list(f.mean(axis=0)) for f in ftrs]
        return ftrs

if __name__ == '__main__':
    dp = DataProcessor('/home/vis/xiaotong/baidu/build/convnet/batches/40w/batches.meta', 16, False)
    cnp = ConvNetPredict('/home/vis/xiaotong/baidu/build/convnet/models/ConvNet__500w-40w-finetune', dp, 3, ['softmax-type', 'softmax-texture', 'softmax-collar', 'softmax-sleeves'])

    I = imread('test.jpg')
    label_names = dp.label_names
    probs = cnp.predict(I)

    for label, prob in zip(label_names, probs):
        for l, p in zip(label, prob):
            print l, p

