from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        super().__init__()
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                in_dim = size_list[i]
                out_dim = size_list[i + 1]
                if act_func == 'ReLU' and i < len(size_list) - 2:
                    scale = np.sqrt(2.0 / in_dim)
                else:
                    scale = np.sqrt(1.0 / in_dim)
                init = lambda size, scale=scale: np.random.normal(0.0, scale, size)
                layer = Linear(in_dim=in_dim, out_dim=out_dim, initialize_method=init)
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplementedError
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(
        self,
        in_channels=1,
        input_size=28,
        conv_channels=4,
        second_conv_channels=8,
        kernel_size=3,
        pool_size=2,
        num_classes=10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.conv_channels = conv_channels
        self.second_conv_channels = second_conv_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.padding = kernel_size // 2
        self.stride = 1

        conv_scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        final_channels = second_conv_channels or conv_channels
        final_size = input_size // pool_size if second_conv_channels is not None else input_size
        second_conv_scale = np.sqrt(2.0 / (conv_channels * kernel_size * kernel_size))
        linear_in = final_channels * final_size * final_size
        linear_scale = np.sqrt(1.0 / linear_in)

        conv_init = lambda size, scale=conv_scale: np.random.normal(0.0, scale, size)
        second_conv_init = lambda size, scale=second_conv_scale: np.random.normal(0.0, scale, size)
        linear_init = lambda size, scale=linear_scale: np.random.normal(0.0, scale, size)

        self.layers = [
            conv2D(
                in_channels=in_channels,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=self.padding,
                initialize_method=conv_init,
            ),
            ReLU(),
        ]
        if second_conv_channels is not None:
            self.layers.extend([
                MaxPool2D(pool_size=pool_size),
                conv2D(
                    in_channels=conv_channels,
                    out_channels=second_conv_channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    initialize_method=second_conv_init,
                ),
                ReLU(),
            ])
        self.layers.extend([
            Flatten(),
            Linear(linear_in, num_classes, initialize_method=linear_init),
        ])

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if X.ndim == 2:
            expected_dim = self.in_channels * self.input_size * self.input_size
            assert X.shape[1] == expected_dim
            X = X.reshape(X.shape[0], self.in_channels, self.input_size, self.input_size)
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)

        config = param_list['config']
        if 'second_conv_channels' not in config and len(param_list['params']) == 2:
            config['second_conv_channels'] = None
        if 'pool_size' not in config:
            config['pool_size'] = 1 if (
                config.get('second_conv_channels') is not None
                and len(param_list['params']) == 3
                and param_list['params'][-1]['W'].shape[0] == config['second_conv_channels'] * config['input_size'] * config['input_size']
            ) else 2
        self.__init__(**config)

        param_idx = 0
        for layer in self.layers:
            if layer.optimizable:
                params = param_list['params'][param_idx]
                layer.W = params['W']
                layer.b = params['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = params['weight_decay']
                layer.weight_decay_lambda = params['lambda']
                param_idx += 1
        
    def save_model(self, save_path):
        param_list = {
            'config': {
                'in_channels': self.in_channels,
                'input_size': self.input_size,
                'conv_channels': self.conv_channels,
                'second_conv_channels': self.second_conv_channels,
                'kernel_size': self.kernel_size,
                'pool_size': self.pool_size,
                'num_classes': self.num_classes,
            },
            'params': [],
        }
        for layer in self.layers:
            if layer.optimizable:
                param_list['params'].append({
                    'W' : layer.params['W'],
                    'b' : layer.params['b'],
                    'weight_decay' : layer.weight_decay,
                    'lambda' : layer.weight_decay_lambda,
                })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
