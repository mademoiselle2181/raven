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
  Created on Jun 30, 2021

  @author: wangc
  One-vs-the-rest (OvR) multiclass strategy classifer
"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from SupervisedLearning.ScikitLearn import ScikitLearnBase
from utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class OneVsRestClassifier(ScikitLearnBase):
  """
    One-vs-the-rest (OvR) multiclass strategy classifer
  """
  info = {'problemtype':'classifer', 'normalize':False}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    import sklearn
    import sklearn.multiclass
    import sklearn.multioutput
    # we wrap the model with the multi output regressor (for multitarget)
    self.model = sklearn.multioutput.MultiOutputClassifier(sklearn.multiclass.OneVsRestClassifier)

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.description = r"""The \xmlNode{OneVsRestClassifier} (\textit{One-vs-the-rest (OvR) multiclass strategy})
                        Also known as one-vs-all, this strategy consists in fitting one classifier per class. For each
                        classifier, the class is fitted against all the other classes. In addition to its computational
                        efficiency (only n\_classes classifiers are needed), one advantage of this approach is its
                        interpretability. Since each class is represented by one and one classifier only, it is
                        possible to gain knowledge about the class by inspecting its corresponding classifier.
                        This is the most commonly used strategy for multiclass classification and is a fair default choice.
                        """
    estimatorInput = InputData.parameterInputFactory("estimator", contentType=InputTypes.StringType,
                                                 descr=r"""An estimator object implementing fit and one of
                                                 decision\_function or predict\_proba.""", default='no-default')
    #TODO: Add more inputspecs for estimator
    specs.addSub(estimatorInput)

    specs.addSub(InputData.parameterInputFactory("n_jobs", contentType=InputTypes.IntegerType,
                                                 descr=r"""TThe number of jobs to use for the computation: the n\_classes one-vs-rest
                                                 problems are computed in parallel. None means 1 unless in a joblib.parallel\_backend
                                                 context. -1 means using all processors.""", default=None))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['estimator','n_jobs'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)