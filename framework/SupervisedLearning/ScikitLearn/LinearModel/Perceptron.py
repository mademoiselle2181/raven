# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
  Created on Jan 21, 2020

  @author: alfoa
  Perceptron Classifier
  
"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .ScikitLearnBase import SciktLearnBase
from utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class Perceptron(SciktLearnBase):
  """
    Perceptron Classifier
  """
  info = {'problemtype':'classification', 'normalize':True}
  
  def __init__(self,messageHandler,**kwargs):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    import sklearn
    import sklearn.linear_model
    import sklearn.multioutput
    # we wrap the model with the multi output regressor (for multitarget)
    self.model = sklearn.multioutput.MultiOutputClassifier(sklearn.linear_model.Perceptron)
    SciktLearnBase.__init__(messageHandler,**kwargs)

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(Perceptron, cls).getInputSpecification()
    specs.description = r"""The \\xmlNode{Perceptron} classifier is based on an
                        algorithm for supervised classification of
                        an input into one of several possible non-binary outputs.
                        It is a type of linear classifier, i.e. a classification algorithm that makes
                        its predictions based on a linear predictor function combining a set of weights
                        with the feature vector.
                        The algorithm allows for online learning, in that it processes elements in the
                        training set one at a time.
                        """
    specs.addSub(InputData.parameterInputFactory("penalty", contentType=InputTypes.makeEnumType("penalty", "penaltyType",['l2', ' l1', 'elasticnet']),
                                                 descr=r"""The penalty (aka regularization term) to be used.""", default=None))
    specs.addSub(InputData.parameterInputFactory("alpha", contentType=InputTypes.FloatType,
                                                 descr=r"""Constant that multiplies the regularization term if regularization is used.""", default=0.0001))
    specs.addSub(InputData.parameterInputFactory("l1_ratio", contentType=InputTypes.FloatType,
                                                 descr=r"""The Elastic Net mixing parameter, with $0 <= l1_ratio <= 1.$ $l1_ratio=0$ corresponds to L2 penalty,
                                                  $l1_ratio=1$ to L1. Only used if penalty='elasticnet'.""", default=0.15))
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether the intercept should be estimated or not. If False,
                                                  the data is assumed to be already centered.""", default=True))
    specs.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.IntegerType,
                                                 descr=r"""The maximum number of passes over the training data (aka epochs).""", default=1000))
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""The stopping criterion.""", default=1e-3))
    specs.addSub(InputData.parameterInputFactory("n_iter_no_change", contentType=InputTypes.IntegerType,
                                                descr=r"""Number of iterations with no improvement to wait before early stopping.""", default=5))
    specs.addSub(InputData.parameterInputFactory("shuffle", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether or not the training data should be shuffled after each epoch.""", default=True))
    specs.addSub(InputData.parameterInputFactory("eta0", contentType=InputTypes.FloatType,
                                                 descr=r"""The stopping criterion.""", default=1))
    specs.addSub(InputData.parameterInputFactory("early_stopping", contentType=InputTypes.BoolType,
                                                 descr=r"""hether to use early stopping to terminate training when validation score is not
                                                 improving. If set to True, it will automatically set aside a stratified fraction of training
                                                 data as validation and terminate training when validation score is not improving by at least
                                                 tol for n_iter_no_change consecutive epochs.""", default=False))
    specs.addSub(InputData.parameterInputFactory("validation_fraction", contentType=InputTypes.FloatType,
                                                 descr=r"""The proportion of training data to set aside as validation set for early stopping.
                                                 Must be between 0 and 1. Only used if early_stopping is True.""", default=0.1))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super(SciktLearnBase, self)._handleInput(self, paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['penalty','alpha','l1_ratio','early_stopping',
                                                               'fit_intercept','max_iter','tol','validation_fraction',
                                                               'n_iter_no_change','shuffle','eta0'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)



