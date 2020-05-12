# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse

import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import flags


class loader:
    def __init__(self,file_name):
        try:
            self.reader=pywrap_tensorflow.NewCheckpointReader(file_name)
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")
            if ("Data loss" in str(e) and
                    any(e in file_name for e in [".index", ".meta", ".data"])):
                proposed_file = ".".join(file_name.split(".")[0:-1])
                v2_file_error_template = """
        It's likely that this is a V2 checkpoint and you need to provide the filename
        *prefix*.  Try removing the '.' and extension.  Try:
        inspect checkpoint --file_name = {}"""
                print(v2_file_error_template.format(proposed_file))

    # ==============================================================================

    def return_tensor_value_list_by_name(self, tensor_list):
        tensor_values=np.array([[self.return_tensor_value_by_name(tens)] for tens in tensor_list])
        return tensor_values

    # ==============================================================================
    def return_tensor_value_by_name(self, tensor_name):
        try:
            if not tensor_name:
                print(self.reader.debug_string().decode("utf-8"))
            else:
                print("tensor_name: ", tensor_name)
                return self.reader.get_tensor(tensor_name)
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))

        return None

    # ==============================================================================
    def print_tensors_in_checkpoint_file(self,file_name, tensor_name, all_tensors,
                                         all_tensor_names=False):
      """Prints tensors in a checkpoint file.
      If no `tensor_name` is provided, prints the tensor names and shapes
      in the checkpoint file.
      If `tensor_name` is provided, prints the content of the tensor.
      Args:
        file_name: Name of the checkpoint file.
        tensor_name: Name of the tensor in the checkpoint file to print.
        all_tensors: Boolean indicating whether to print all tensors.
        all_tensor_names: Boolean indicating whether to print all tensor names.
      """
      try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors or all_tensor_names:
          var_to_shape_map = reader.get_variable_to_shape_map()
          for key in sorted(var_to_shape_map):
            print("tensor_name: ", key)
            if all_tensors:
              print(reader.get_tensor(key))
        elif not tensor_name:
          print(reader.debug_string().decode("utf-8"))
        else:
          print("tensor_name: ", tensor_name)
          print(reader.get_tensor(tensor_name))
      except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
          print("It's likely that your checkpoint file has been compressed "
                "with SNAPPY.")
        if ("Data loss" in str(e) and
            any(e in file_name for e in [".index", ".meta", ".data"])):
          proposed_file = ".".join(file_name.split(".")[0:-1])
          v2_file_error_template = """
    It's likely that this is a V2 checkpoint and you need to provide the filename
    *prefix*.  Try removing the '.' and extension.  Try:
    inspect checkpoint --file_name = {}"""
          print(v2_file_error_template.format(proposed_file))

    # ==============================================================================
    def parse_numpy_printoption(self,kv_str):
      """Sets a single numpy printoption from a string of the form 'x=y'.
      See documentation on numpy.set_printoptions() for details about what values
      x and y can take. x can be any option listed there other than 'formatter'.
      Args:
        kv_str: A string of the form 'x=y', such as 'threshold=100000'
      Raises:
        argparse.ArgumentTypeError: If the string couldn't be used to set any
            nump printoption.
      """
      k_v_str = kv_str.split("=", 1)
      if len(k_v_str) != 2 or not k_v_str[0]:
        raise argparse.ArgumentTypeError("'%s' is not in the form k=v." % kv_str)
      k, v_str = k_v_str
      printoptions = np.get_printoptions()
      if k not in printoptions:
        raise argparse.ArgumentTypeError("'%s' is not a valid printoption." % k)
      v_type = type(printoptions[k])
      if v_type is type(None):
        raise argparse.ArgumentTypeError(
            "Setting '%s' from the command line is not supported." % k)
      try:
        v = (
            v_type(v_str)
            if v_type is not bool else flags.BooleanParser().parse(v_str))
      except ValueError as e:
        raise argparse.ArgumentTypeError(e.message)
      np.set_printoptions(**{k: v})

