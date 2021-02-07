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
  The ScikitLearn folder includes different types of ScikitLearn based ROMs
  that are available via RAVEN

  Created on Jan 21, 2020
  @author: alfoa
"""
# These lines ensure that we do not have to do something like:
# 'from Samplers.Sampler import Sampler' outside of this submodule
from .SciktLearnBase import SciktLearnBase
# Factory methods
#from .Factory import knownTypes
#from .Factory import returnInstance
#from .Factory import returnClass