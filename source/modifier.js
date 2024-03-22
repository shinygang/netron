var modifier = modifier || {};

modifier.Modifier = class {
  constructor(view) {
    this.view = view;
    this.model = null;
    this.stack = null;
    this.name2ModelNode = new Map();
    this.name2ViewNode = new Map();
    this.name2NodeStates = new Map();
    this.namedEdges = new Map();

    this.addedOutputs = new Set();
    this.addedInputs = new Map();
    this.deleteInputs = new Map();
    this.addedNode = new Map();
    this.addNodeKey = 0;
    this.changedAttributes = new Map();
    this.addAttributes = new Map();
    this.delAttributes = new Map();
    this.initializerEditInfo = new Map();
    this.renameMap = new Map();
    this.renameTypeMap = new Map();
    this.reBatchInfo = new Map();
    // 记录操作步骤
    this.operatingSteps = new Set();

    this.downloadWithShapeInf = false;
    this.downloadWithCleanUp = false;
  }

  loadModelGraph(model, stack) {
    this.model = model;
    this.stack = stack;
    this.graph = stack[0].graph;
    // this.analyzeModelGraph();

    this.updateAddNodeDropDown();
  }

  // TODO: add filter feature like here: https://www.w3schools.com/howto/howto_js_dropdown.asp
  updateAddNodeDropDown() {
    // update dropdown supported node lost
    var addNodeDropdown = this.view._host.document.getElementById("add-node-dropdown");
    for (const node of this.model.supported_nodes) {
      // node: [domain, op]
      var option = new Option(node[1], node[0] + ":" + node[1]);
      // console.log(option)
      addNodeDropdown.appendChild(option);
    }
  }

  getShapeInfo(name) {
    for (var value_info of this.graph._value_info) {
      if (value_info.name == name && value_info.type && value_info.type.tensor_type) {
        var tensor_type = value_info.type.tensor_type;
        let shape = [];
        if (tensor_type.shape && tensor_type.shape.dim) {
          shape = tensor_type.shape.dim.map((dim) =>
            dim.dim_param ? dim.dim_param : dim.dim_value ? dim.dim_value : null
          );
        }
        var shape_info = this.graph._context
          .createTensorType(tensor_type.elem_type, shape)
          .toString();
        return shape_info;
      }
      break;
    }
    return null;
  }

  // ======= Record modified info =======> //
  addNode(op_domain, op_type) {
    var node_id = (this.addNodeKey++).toString(); // in case input (onnx) node has no name
    var modelNodeName = "custom_added_" + op_type + node_id;

    var properties = new Map();
    properties.set("domain", op_domain);
    properties.set("op_type", op_type);
    properties.set("name", modelNodeName);
    this.addedNode.set(modelNodeName, this.view._newLightNode(properties));

    this.operatingSteps.add({
      type: 'addNode',
      modelNodeName: modelNodeName,
    })

    this.applyAndUpdateView();
  }

  addModelOutput(node_name) {
    var modelNode = this.name2ModelNode.get(node_name);
    // use a output argument as a proxy
    var output_name = modelNode.outputs[0].value[0].name;
    if (this.name2NodeStates.get("out_" + output_name)) {
      this.recoverSingleNode("out_" + output_name);
    } else {
      this.addedOutputs.add(output_name);
    }

    this.operatingSteps.add({
      type: 'addOutput',
      modelNodeName: modelNode,
      value: {
        output_name
      }
    })

    this.applyAndUpdateView();
  }

  addModelInput(modelNodeName, input_name, input_shape_type) {
    const inputItem = this.graph.add_input([input_name, input_shape_type]);
    this.addedInputs.set(modelNodeName, inputItem);

    this.operatingSteps.add({
      type: 'addInput',
      modelNodeName,
      value: {
        input_name
      }
    })

    this.applyAndUpdateView();
    this.addedInputs = new Map();

    
  }

  deleteInput(modelNodeName, input_name) {
    if (!this.deleteInput.get(modelNodeName)) {
      this.deleteInput.set(modelNodeName, new Map());
    }
    this.deleteInput.get(modelNodeName).set(input_name, true);

    this.operatingSteps.add({
      type: 'deleteInput',
      modelNodeName,
      value: {
        input_name
      }
    })

    // this.view._updateGraph()
    this.applyAndUpdateView();
  }

  deleteModelOutput(output_name) {
    this.name2NodeStates.set(output_name, "Deleted"); // "out_" + xxx
    this.operatingSteps.add({
      type: 'deleteOutput',
      value: {
        output_name
      }
    })

    this.applyAndUpdateView();
  }

  deleteSingleNode(node_name) {
    this.name2NodeStates.set(node_name, "Deleted");
    this.name2ViewNode.get(node_name).element.style.opacity = 0.3;

    this.operatingSteps.add({
      type: 'deleteNode',
      value: {
        node_name
      }
    })
  }

  deleteNodeWithChildren(node_name) {
    if (this.name2NodeStates.get(node_name) == "Deleted") return;

    this.operatingSteps.add({
      type: 'deleteNode',
      value: {
        node_name
      }
    })

    this.name2NodeStates.set(node_name, "Deleted");
    this.name2ViewNode.get(node_name).element.style.opacity = 0.3;

    if (!this.namedEdges.has(node_name)) return; // for leaf node

    for (var i = 0; i < this.namedEdges.get(node_name).length; i++) {
      this.deleteNodeWithChildren(this.namedEdges.get(node_name)[i]);
    }
  }

  recoverSingleNode(node_name) {
    this.name2NodeStates.set(node_name, "Exist");
    this.name2ViewNode.get(node_name).element.style.opacity = 1;
  }

  recoverNodeWithChildren(node_name) {
    if (this.name2NodeStates.get(node_name) == "Exist") return;

    this.name2NodeStates.set(node_name, "Exist");
    this.name2ViewNode.get(node_name).element.style.opacity = 1;

    if (!this.namedEdges.has(node_name)) return; // for leaf node

    for (var i = 0; i < this.namedEdges.get(node_name).length; i++) {
      this.recoverNodeWithChildren(this.namedEdges.get(node_name)[i]);
    }
  }

  getOriginalName(param_type, modelNodeName, param_index, arg_index) {
    if (param_type == "model_input") {
      var orig_arg_name = this.name2ModelNode.get(modelNodeName).value[0].original_name;
    }

    if (param_type == "model_output") {
      // modelNodeName = 'out_' + modelNodeName
      // console.log(modelNodeName)
      var orig_arg_name = this.name2ModelNode.get(modelNodeName).value[0].original_name;
      // console.log(orig_arg_name)
    }

    if (param_type == "input") {
      var orig_arg_name =
        this.name2ModelNode.get(modelNodeName).inputs[param_index].value[arg_index].original_name;
      // console.log(orig_arg_name)
    }
    if (param_type == "output") {
      var orig_arg_name =
        this.name2ModelNode.get(modelNodeName).outputs[param_index].value[arg_index].original_name;
      // console.log(orig_arg_name)
    }

    return orig_arg_name;
  }

  changeNodeInputOutput(
    modelNodeName,
    parameterName,
    param_type,
    param_index,
    arg_index,
    targetValue
  ) {
    if (this.addedNode.has(modelNodeName)) {
      // for custom added node
      if (this.addedNode.get(modelNodeName).inputs.has(parameterName)) {
        var arg_name = this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0]; // [arg.name, arg.is_optional]
        // update the corresponding initializer name
        if (this.initializerEditInfo.has(arg_name)) {
          var init_val = this.initializerEditInfo.get(arg_name);
          this.initializerEditInfo.set(targetValue, init_val);
          this.initializerEditInfo.delete(arg_name);
        }
        this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0] = targetValue;
      }
      // console.log(this.initializerEditInfo)

      if (this.addedNode.get(modelNodeName).outputs.has(parameterName)) {
        this.addedNode.get(modelNodeName).outputs.get(parameterName)[arg_index][0] = targetValue;
      }
    }
    // console.log(this.addedNode)
    else {
      // for the nodes in the original model
      var orig_arg_name = this.getOriginalName(param_type, modelNodeName, param_index, arg_index);
      // console.log(orig_arg_name)

      if (!this.renameMap.get(modelNodeName)) {
        this.renameMap.set(modelNodeName, new Map());
      }
      this.renameMap.get(modelNodeName).set(orig_arg_name, targetValue);

      this.operatingSteps.add({
        type: 'changeNodeInputOutput',
        modelNodeName,
        value: {
          input_name: param_type === 'input' ? orig_arg_name : '',
          output_name: param_type === 'output' ? orig_arg_name : '',
        }
      })
    }

    
    // this.view._updateGraph()

    this.applyAndUpdateView();
  }

  changeNodeInputOutputType(
    modelNodeName,
    param_type,
    param_index,
    arg_index,
    targetValue
  ) {
    var orig_arg_name = this.getOriginalName(param_type, modelNodeName, param_index, arg_index);

    if (!this.renameTypeMap.get(modelNodeName)) {
      this.renameTypeMap.set(modelNodeName, new Map());
    }
    this.renameTypeMap.get(modelNodeName).set(orig_arg_name, targetValue);
    // this.view._updateGraph()

    this.applyAndUpdateView();
  }

  delNodeInput(modelNodeName, parameterName) {
    if (!this.deleteInputs.get(modelNodeName)) {
      this.deleteInputs.set(modelNodeName, new Map());
    }
    this.deleteInputs.get(modelNodeName).set(parameterName, true);
    this.applyAndUpdateView();
  }

  changeInitializer(
    modelNodeName,
    parameterName,
    param_type,
    param_index,
    arg_index,
    type,
    targetValue
  ) {
    var orig_arg_name = this.getOriginalName(param_type, modelNodeName, param_index, arg_index);
    this.initializerEditInfo.set(orig_arg_name, [type, targetValue]);
    // this.view._updateGraph()

    this.applyAndUpdateView();
  }

  changeAddedNodeInitializer(
    modelNodeName,
    parameterName,
    param_type,
    param_index,
    arg_index,
    type,
    targetValue
  ) {
    var arg_name = this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0];
    this.initializerEditInfo.set(arg_name, [type, targetValue]);
    // this.view._updateGraph()

    this.applyAndUpdateView();
  }

  changeNodeAttribute(modelNodeName, attributeName, targetValue, type) {
    if (this.addedNode.has(modelNodeName)) {
      this.addedNode.get(modelNodeName).attributes.set(attributeName, [targetValue, type]);
    }
    // console.log(this._addedNode)
      // for the nodes in the original model
    if (!this.changedAttributes.get(modelNodeName)) {
      this.changedAttributes.set(modelNodeName, new Map());
    }
    this.changedAttributes.get(modelNodeName).set(attributeName, [targetValue, type]);

    // this.view._updateGraph()
    this.applyAndUpdateView();
  }

  delNodeAttribute(modelNodeName, attributeName) {
    if (!this.delAttributes.get(modelNodeName)) {
      this.delAttributes.set(modelNodeName, new Map());
    }
    this.delAttributes.get(modelNodeName).set(attributeName, true);
    // this.view._updateGraph()
    this.applyAndUpdateView();
  }

  addNodeAttribute(modelNodeName, attributeName, type, targetValue) {
    if (this.addedNode.has(modelNodeName)) {
        this.addedNode.get(modelNodeName).attributes.set(attributeName, [targetValue, type]);
    }
    if (!this.addAttributes.get(modelNodeName)) {
      this.addAttributes.set(modelNodeName, new Map());
    }
    this.addAttributes.get(modelNodeName).set(attributeName, [targetValue, type]);
    this.applyAndUpdateView();
  }

  changeBatchSize(type, value) {
    if (type === "fixed") {
      this.reBatchInfo.set("type", "fixed");
      this.reBatchInfo.set("value", value);
    } else {
      // dynamic
      this.reBatchInfo.set("type", "dynamic");
      this.reBatchInfo.set("value", "dynamic");
    }
  }

  onOffShapeInf(turnedOn) {
    if (turnedOn) this.downloadWithShapeInf = true;
    else this.downloadWithShapeInf = false;
  }

  onOffCleanUp(turnedOn) {
    if (turnedOn) this.downloadWithCleanUp = true;
    else this.downloadWithCleanUp = false;
  }
  // <======= Record modified info ======= //

  // ======= Apply modified info and update view =======> //
  deleteEnter() {
    this.applyAndUpdateView();
  }

  refreshModelInputOutput() {
    this.graph.reset_custom_modified_inputs();
    for (var node of this.graph._nodes) {
        if (this.addedInputs.get(node.modelNodeName)) {
            node.addInput(this.addedInputs.get(node.modelNodeName))
        }
        if (this.deleteInputs.get(node.modelNodeName)) {
          for (var [key] of this.deleteInputs.get(node.modelNodeName)) {
            node.deleteInput(key)
          }
        }
    }
    for (var input of this.graph.inputs) {
      var input_orig_name = input.value[0].original_name;
      console.log(input_orig_name)
      if (this.renameMap.get(input_orig_name)) {
        var new_name = this.renameMap.get(input_orig_name).get(input_orig_name);
        var arg_with_new_name = this.graph._context.value(new_name, input_orig_name);

        input.value[0] = arg_with_new_name;

        // change all the name of node input linked with model input meanwhile
        for (var node of this.graph.nodes) {
          for (var node_input of node.inputs) {
            for (const [index, element] of node_input.value.entries()) {
              if (element.original_name == input_orig_name) {
                var arg_with_new_name = this.graph._context.value(new_name, element.original_name);

                node_input.value[index] = arg_with_new_name;

                // save the changed name into _renameMap
                // as this modified _renamedMap, so refreshModelInputOutput() shoulf be called before refreshNodeArguments()
                if (!this.renameMap.get(node.modelNodeName)) {
                  this.renameMap.set(node.modelNodeName, new Map());
                }

                var orig_arg_name = element.original_name;
                this.renameMap.get(node.modelNodeName).set(orig_arg_name, new_name);
              }
            }
          }
        }
      }
    }
    // console.log(this.graph.outputs)
    // create and add new output to graph
    this.graph.reset_custom_modified_outputs();
    for (var output_name of this.addedOutputs) {
      this.graph.add_output(output_name);
    }
    for (var output of this.graph.outputs) {
      var output_orig_name = output.value[0].original_name;
      if (this.renameMap.get("out_" + output_orig_name)) {
        // for model input and output, node.modelNodeName == element.name
        var new_name = this.renameMap.get("out_" + output_orig_name).get(output_orig_name);
        // console.log(new_name)
        var arg_with_new_name = this.graph._context.value(new_name, output_orig_name);

        output.value[0] = arg_with_new_name;

        // change all the name of node output linked with the model output meanwhile
        for (var node of this.graph.nodes) {
          for (var node_output of node.outputs) {
            for (const [index, element] of node_output.value.entries()) {
              if (element.original_name == output_orig_name) {
                // console.log(element.name)
                var arg_with_new_name = this.graph._context.value(new_name, element.original_name);

                node_output.value[index] = arg_with_new_name;

                // save the changed name into _renameMap
                // as this modified _renamedMap, so refreshModelInputOutput() shoulf be called before refreshNodeArguments()
                if (!this.renameMap.get(node.modelNodeName)) {
                  this.renameMap.set(node.modelNodeName, new Map());
                }

                var orig_arg_name = element.original_name;
                this.renameMap.get(node.modelNodeName).set(orig_arg_name, new_name);
              }
            }
          }
        }
      }
    }

    for (var output of this.graph.outputs) {
      var output_orig_name = output.value[0].original_name;
      if (this.name2NodeStates.get("out_" + output_orig_name) == "Deleted") {
        this.graph.delete_output(output_orig_name);
      }
    }
  }

  // re-generate the added node according to addedNode according to the latest addedNode
  refreshAddedNode() {
    this.graph.reset_custom_added_node();
    // for (const node_info of this.addedNode.values()) {
    // for (const [modelNodeName, node_info] of this.lastViewGraph.addedNode) {
    for (const [modelNodeName, node_info] of this.addedNode) {
      var node = this.graph.make_custom_added_node(node_info);

      for (const input of node.inputs) {
        var arg_list_info = [];
        for (const arg of input.value) {
          arg_list_info.push([arg.name, arg.is_optional]);
        }
        this.addedNode.get(modelNodeName).inputs.set(input.name, arg_list_info);
      }

      for (const output of node.outputs) {
        var arg_list_info = [];
        for (const arg of output.value) {
          arg_list_info.push([arg.name, arg.is_optional]);
        }
        this.addedNode.get(modelNodeName).outputs.set(output.name, arg_list_info);
      }
    }
  }

  // re-fresh node arguments in case the node inputs/outputs are changed
  refreshNodeArguments() {
    for (var input of this.graph._inputs) {
      if (this.renameTypeMap.get(input.modelNodeName)) {
        for (const [index, element] of input.value.entries()) {
          if (
            this.renameTypeMap.get(input.modelNodeName) &&
            this.renameTypeMap.get(input.modelNodeName).get(element.original_name)
          ) {
            // 修改顶部节点input的type
            var dataType = this.renameTypeMap.get(input.modelNodeName).get(element.original_name);
            if (dataType.indexOf("[") >= 0) {
              input.value[index]._type._dataType = dataType.split("[")[0];
              input.value[index]._type._shape._dimensions = dataType
                .split("[")[1]
                .replace("]", "")
                .split(",")
                .map((cur) => {
                  try {
                    return BigInt(cur);
                  } catch (error) {
                    return cur;
                  }
                });
            } else {
              input.value[index]._type._dataType = dataType;
            }
          }
        }
      }
    }
    for (var node of this.graph.nodes) {
      // if (this.modifier.renameMap.get(node.modelNodeName)) {
      if (this.renameMap.get(node.modelNodeName)) {
        // check inputs
        for (var input of node.inputs) {
          for (const [index, element] of input.value.entries()) {
            if (
              this.renameMap.get(node.modelNodeName) &&
              this.renameMap.get(node.modelNodeName).get(element.original_name)
            ) {
              var new_name = this.renameMap.get(node.modelNodeName).get(element.original_name);
              var arg_with_new_name = this.graph._context.value(new_name, element.original_name);

              input.value[index] = arg_with_new_name;
            }
          }
        }

        // check outputs
        for (var output of node.outputs) {
          for (const [index, element] of output.value.entries()) {
            if (
              this.renameMap.get(node.modelNodeName) &&
              this.renameMap.get(node.modelNodeName).get(element.original_name)
            ) {
              var new_name = this.renameMap.get(node.modelNodeName).get(element.original_name);
              // console.log(new_name)
              var arg_with_new_name = this.graph._context.value(new_name, element.original_name);

              output.value[index] = arg_with_new_name;
            }
          }
        }
      }
    }

    this.renameTypeMap = new Map();
    this.renameMap = new Map();
    this.namedEdges = new Map();
  }

  refreshNodeAttributes() {
    // 修改的属性
    for (const node_name of this.changedAttributes.keys()) {
      var attr_change_map = this.changedAttributes.get(node_name);
      var node = this.name2ModelNode.get(node_name);
      for (var i = 0; i < node.attributes.length; ++i) {
        if (attr_change_map.get(node.attributes[i].name)) {
          // [val, type]
          node.attributes[i].value = attr_change_map.get(node.attributes[i].name)[0];
        }
      }
    }

    // 添加的属性
    for (const node_name of this.addAttributes.keys()) {
        var attr_change_map = this.addAttributes.get(node_name);
        var node = this.name2ModelNode.get(node_name);
        for (const attr_name of attr_change_map.keys()) {
            const [targetValue, type] = attr_change_map.get(attr_name);
            node.addAttribute(attr_name, type,  targetValue);
        }
    }

    // 删除的属性
    for (const node_name of this.delAttributes.keys()) {
      var attr_del_map = this.delAttributes.get(node_name);
      var node = this.name2ModelNode.get(node_name);
      for (var i = 0; i < node.attributes.length; ++i) {
        if (attr_del_map.has(node.attributes[i].name)) {
          node.deleteAttribute(node.attributes[i].name);
        }
      }
    }

    // this.changedAttributes = new Map();
    this.addAttributes = new Map();
    // this.delAttributes = new Map();
  }

  resetGraph() {
    // reset node states
    for (const name of this.name2NodeStates.keys()) {
      this.name2NodeStates.set(name, "Exist");
    }

    // console.log(this.modifier.renameMap)
    // reset node inputs/outputs
    for (const changed_node_name of this.renameMap.keys()) {
      var node = this.name2ModelNode.get(changed_node_name);
      // console.log(node)
      // console.log(typeof node)
      // console.log(node.constructor.name)
      if (node.value) {
        // model input or model output. Because they are purely onnx.Parameter
        // node.value[0] = this.graph._context.value(node.modelNodeName);
        node.value[0] = this.graph._context.value(node.value[0].original_name);
      } else {
        // model nodes
        //reset inputs
        for (var input of node.inputs) {
          for (var i = 0; i < input.value.length; ++i) {
            // console.log(input.value[i].name)
            if (this.renameMap.get(node.modelNodeName).get(input.value[i].original_name)) {
              input.value[i] = this.graph._context.value(input.value[i].original_name);
            }
          }
        }

        // reset outputs
        for (var output of node.outputs) {
          for (var i = 0; i < output.value.length; ++i) {
            if (this.renameMap.get(node.modelNodeName).get(output.value[i].original_name)) {
              output.value[i] = this.graph._context.value(output.value[i].original_name);
            }
          }
        }
      }
    }
    this.namedEdges = new Map();
    this.changedAttributes = new Map();
    this.addAttributes = new Map();
    this.delAttributes = new Map();
    this.initializerEditInfo = new Map();
    this.renameMap = new Map();
    this.reBatchInfo = new Map();

    // clear custom added nodes
    this.addedNode = new Map();
    this.graph.reset_custom_added_node();
    this.addedOutputs = new Set();
    this.graph.reset_custom_modified_outputs();
    this.addedInputs = new Map();
    this.deleteInputs = new Map();
    this.graph.reset_custom_modified_inputs();

    // reset load location
    var container = this.view._element("graph");
    container.scrollLeft = 0;
    container.scrollTop = 0;
    this.view._zoom = 1;

    this.applyAndUpdateView();
  }

  applyAndUpdateView() {
    this.refreshAddedNode();
    this.refreshModelInputOutput();
    this.refreshNodeArguments();
    this.refreshNodeAttributes();

    // this.graphs has been modified (inplace)
    console.log("this.model, this.stack:", this.model, this.stack);
    this.view._updateGraph(this.model, this.stack);
  }
  // <======= Apply modified info and update view ======= //
};

if (typeof module !== "undefined" && typeof module.exports === "object") {
  module.exports.Modifier = modifier.Modifier;
}
