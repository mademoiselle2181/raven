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
  Least Angle Regression model
  
"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
from numpy import finfo
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .ScikitLearnBase import SciktLearnBase
from utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class LarsCV(SciktLearnBase):
  """
    Cross-validated Least Angle Regression model.
  """
  info = {'problemtype':'regression', 'normalize':False}
  
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
    self.model = sklearn.multioutput.MultiOutputRegressor(sklearn.linear_model.LarsCV)
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
    specs = super(LarsCV, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{LarsCV} is Cross-validated \textit{Least Angle Regression model} model
                        is a regression algorithm for high-dimensional data.
                        The LARS algorithm provides a means of producing an estimate of which variables
                        to include, as well as their coefficients, when a response variable is
                        determined by a linear combination of a subset of potential covariates.
                        This method is an augmentation of the Lars method with the addition of cross-validation
                        embedded tecniques.
                        """
    specs.addSub(InputData.parameterInputFactory("eps", contentType=InputTypes.FloatType,
                                                 descr=r"""The machine-precision regularization in the computation of the Cholesky
                                                 diagonal factors. Increase this for very ill-conditioned systems. Unlike the tol
                                                 parameter in some iterative optimization-based algorithms, this parameter does not
                                                 control the tolerance of the optimization.""", default=finfo(float).eps))
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether the intercept should be estimated or not. If False,
                                                  the data is assumed to be already centered.""", default=True))
    specs.addSub(InputData.parameterInputFactory("precompute", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether to use a precomputed Gram matrix to speed up calculations.
                                                 For sparse input this option is always True to preserve sparsity.""", default=False)
    specs.addSub(InputData.parameterInputFactory("normalize", contentType=InputTypes.BoolType,
                                                 descr=r"""This parameter is ignored when fit_intercept is set to False. If True,
                                                 the regressors X will be normalized before regression by subtracting the mean and
                                                 dividing by the l2-norm.""", default=True))
    specs.addSub(InputData.parameterInputFactory("max_n_alphas", contentType=InputTypes.IntegerType,
                                                 descr=r"""The maximum number of points on the path used to compute the
                                                 residuals in the cross-validation.""", default=1000))
    specs.addSub(InputData.parameterInputFactory("cv", contentType=InputTypes.IntegerType,
                                                 descr=r"""Determines the cross-validation splitting strategy.
                                                 It specifies the number of folds..""", default=5)

    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super(SciktLearnBase, self)._handleInput(self, paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['eps','precompute', 'fit_intercept',
                                                               'normalize','max_n_alphas','cv'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)



