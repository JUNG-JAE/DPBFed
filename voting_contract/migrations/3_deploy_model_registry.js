const ModelRegistry = artifacts.require("ModelRegistry");

module.exports = function (deployer) {
  deployer.deploy(ModelRegistry);
};
