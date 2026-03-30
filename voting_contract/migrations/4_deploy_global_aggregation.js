const GlobalAggregation = artifacts.require("GlobalAggregation");

module.exports = function (deployer) {
  // totalShards: 2, alpha: 1/2 = 0.5 (parameter.py의 GLOBAL_MODEL_ALPHA = 0.5)
  const totalShards = 2;
  const alphaNumerator = 1;
  const alphaDenominator = 2;
  deployer.deploy(GlobalAggregation, totalShards, alphaNumerator, alphaDenominator);
};
