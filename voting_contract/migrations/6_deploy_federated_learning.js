const FederatedLearning = artifacts.require("FederatedLearning");

module.exports = function (deployer) {
  // totalShards: 2, alpha: 1/2 = 0.5
  const totalShards = 2;
  const alphaNumerator = 1;
  const alphaDenominator = 2;
  deployer.deploy(FederatedLearning, totalShards, alphaNumerator, alphaDenominator);
};
