from layers import Linear, ReLU, dropout
from classifiers import LogisticRegression
from collections import OrderedDict
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared


def build_shared_zeros(shape, name):
    return shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
            name=name, borrow=True)


class NeuralNet(object):  # TODO refactor with a base class for this and AB
    def __init__(self, numpy_rng, theano_rng=None, 
            n_ins=40*3,
            layers_types=[Linear, ReLU, ReLU, ReLU, LogisticRegression],
            layers_sizes=[1024, 1024, 1024, 1024],
            n_outs=62 * 3,
            rho=0.8, eps=1.5E-7,  # TODO refine
            debugprint=False):
        self.layers = []
        self.params = []
        self.n_layers = len(layers_types)
        self.layers_types = layers_types
        assert self.n_layers > 0
        self._rho = rho  # ``momentum'' for adadelta
        self._eps = eps  # epsilon for adadelta
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta

        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.fmatrix('x')
        self.y = T.ivector('y')
        
        self.layers_ins = [n_ins] + layers_sizes
        self.layers_outs = layers_sizes + [n_outs]
        
        layer_input = self.x
        
        for layer_type, n_in, n_out in zip(layers_types,
                self.layers_ins, self.layers_outs):
            this_layer = layer_type(rng=numpy_rng,
                    input=layer_input, n_in=n_in, n_out=n_out)
            assert hasattr(this_layer, 'output')
            self.params.extend(this_layer.params)
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer.params])
            self.layers.append(this_layer)
            layer_input = this_layer.output

        assert hasattr(self.layers[-1], 'training_cost')
        assert hasattr(self.layers[-1], 'errors')
        # TODO standardize cost
        self.mean_cost = self.layers[-1].negative_log_likelihood(self.y)
        self.cost = self.layers[-1].training_cost(self.y)
        if debugprint:
            theano.printing.debugprint(self.cost)

        self.errors = self.layers[-1].errors(self.y)

    def __repr__(self):
        dimensions_layers_str = map(lambda x: "x".join(map(str, x)),
                zip(self.layers_ins, self.layers_outs))
        return "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
            zip(self.layers_types, dimensions_layers_str)))


    def get_SGD_trainer(self):
        """ Returns a plain SGD minibatch trainer with learning rate as param.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        # using mean_cost so that the learning rate is not too dependent on the batch size
        gparams = T.grad(self.mean_cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * learning_rate 

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=self.mean_cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def get_adadelta_trainer(self):
        """ Returns an Adadelta (Zeiler 2012) trainer using self._rho and 
        self._eps params.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads,
                self._accudeltas, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps) / (agrad + self._eps)) * gparam
            updates[accudelta] = self._rho * accudelta + (1 - self._rho) * dx * dx
            updates[param] = param + dx
            updates[accugrad] = agrad

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y)],
            outputs=self.cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def get_adagrad_trainer(self):
        """ Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, param, gparam in zip(self._accugrads, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = accugrad + gparam * gparam
            dx = - (learning_rate / T.sqrt(agrad + self._eps)) * gparam
            updates[param] = param + dx
            updates[accugrad] = agrad

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=self.cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def score_classif(self, given_set):
        """ Returns functions to get current classification scores. """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        score = theano.function(inputs=[theano.Param(batch_x), theano.Param(batch_y)],
                outputs=self.errors,
                givens={self.x: batch_x, self.y: batch_y})

        # Create a function that scans the entire set given as input
        def scoref():
            return [score(batch_x, batch_y) for batch_x, batch_y in given_set]

        return scoref


class DropoutNet(NeuralNet):
    def __init__(self, numpy_rng, theano_rng=None, 
            n_ins=40*3,
            layers_types=[Linear, ReLU, ReLU, ReLU, LogisticRegression],
            layers_sizes=[1024, 1024, 1024, 1024],
            dropout_rates=[0.2, 0.5, 0.5, 0.5, 0.5],
            n_outs=62 * 3,
            rho=0.9, eps=1.E-6,  # TODO refine
            debugprint=False):
        super(DropoutNet, self).__init__(numpy_rng, theano_rng, n_ins,
                layers_types, layers_sizes, n_outs, rho, eps, debugprint)

        self.dropout_rates = dropout_rates
        dropout_layer_input = dropout(numpy_rng, self.x, p=dropout_rates[0])
        self.dropout_layers = []

        for layer, layer_type, n_in, n_out, dr in zip(self.layers,
                layers_types, self.layers_ins, self.layers_outs,
                dropout_rates[1:] + [0]):  # !!! we do not dropout anything 
                                            # from the last layer !!!
            this_layer = layer_type(rng=numpy_rng,
                    input=dropout_layer_input, n_in=n_in, n_out=n_out,
                    W=layer.W * 1. / (1. - dr), # experimental
                    b=layer.b * 1. / (1. - dr)) # TODO check
            assert hasattr(this_layer, 'output')
            # N.B. dropout with dr=1 does not dropanything!!
            this_layer.output = dropout(numpy_rng, this_layer.output, dr)
            self.dropout_layers.append(this_layer)
            dropout_layer_input = this_layer.output

        assert hasattr(self.layers[-1], 'training_cost')
        assert hasattr(self.layers[-1], 'errors')
        # TODO standardize cost
        # these are the dropout costs
        self.mean_cost = self.dropout_layers[-1].negative_log_likelihood(self.y)
        self.cost = self.dropout_layers[-1].training_cost(self.y)

        # these is the non-dropout errors
        self.errors = self.layers[-1].errors(self.y)

    def __repr__(self):
        return super(DropoutNet, self).__repr__() + "\n"\
                + "dropout rates: " + str(self.dropout_rates)


class DatasetMiniBatchIterator(object):
    def __init__(self, x, y, batch_size=100):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        for i in xrange((self.x.shape[0]+self.batch_size-1)
                / self.batch_size):
            yield (self.x[i*self.batch_size:(i+1)*self.batch_size],
                   self.y[i*self.batch_size:(i+1)*self.batch_size])


def add_fit_and_score(class_to_chg):
    from types import MethodType
    def fit(self, x_train, y_train, x_dev=None, y_dev=None,
            max_epochs=100, early_stopping=True, split_ratio=0.1):
        import time, copy
        if x_dev == None or y_dev == None:
            from sklearn.cross_validation import train_test_split
            x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                    test_size=split_ratio, random_state=42)
        train_fn = self.get_adadelta_trainer()
        train_set_iterator = DatasetMiniBatchIterator(x_train, y_train)
        dev_set_iterator = DatasetMiniBatchIterator(x_dev, y_dev)
        train_scoref = self.score_classif(train_set_iterator)
        dev_scoref = self.score_classif(dev_set_iterator)
        best_dev_loss = numpy.inf
        epoch = 0
        # TODO early stopping
        while epoch < max_epochs:
            avg_costs = []
            timer = time.time()
            for x, y in train_set_iterator:
                avg_cost = train_fn(x, y)
                if type(avg_cost) == list:
                    avg_costs.append(avg_cost[0])
                else:
                    avg_costs.append(avg_cost)
            print('  epoch %i took %f seconds' % (epoch, time.time() - timer))
            print('  epoch %i, avg costs %f' % \
                  (epoch, numpy.mean(avg_costs)))
            print('  epoch %i, training error %f' % \
                  (epoch, numpy.mean(train_scoref())))
            dev_errors = numpy.mean(dev_scoref())
            if dev_errors < best_dev_loss:
                best_dev_loss = dev_errors
                best_params = copy.deepcopy(self.params)
                print('!!!  epoch %i, validation error of best model %f' % \
                      (epoch, dev_errors))
            epoch += 1
        for i, param in enumerate(best_params):
            self.params[i] = param

    def score(self, x, y):
        it = DatasetMiniBatchIterator(x, y)
        scoref = self.score_classif(it)
        return numpy.mean(scoref())

    class_to_chg.fit = MethodType(fit, None, class_to_chg)
    class_to_chg.score = MethodType(score, None, class_to_chg)


class RegularizedNet(NeuralNet):
    def __init__(self, numpy_rng, theano_rng=None, 
            n_ins=100,
            layers_types=[ReLU, ReLU, ReLU, LogisticRegression],
            layers_sizes=[1024, 1024, 1024],
            n_outs=2,
            rho=0.9, eps=1.E-6,  # TODO refine
            L1_reg=0.,
            L2_reg=0.,
            debugprint=False):
        super(EasyNet, self).__init__(numpy_rng, theano_rng, n_ins,
                layers_types, layers_sizes, n_outs, rho, eps, debugprint)

        L1 = shared(0.)
        for param in self.params:
            L1 += T.sum(abs(param)) #/ param.shape[0]
        if L1_reg > 0.:
            self.cost = self.cost + L1_reg * L1
        L2 = shared(0.)
        for param in self.params:
            L2 += T.sum(param ** 2) #/ param.shape[0]
        if L2_reg > 0.:
            self.cost = self.cost + L2_reg * L2

