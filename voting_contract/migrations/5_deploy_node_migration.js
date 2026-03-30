const NodeMigration = artifacts.require("NodeMigration");

module.exports = function (deployer) {
  // totalShards: 2 (parameter.py의 SHARD_NUM = 2)
  const totalShards = 2;
  deployer.deploy(NodeMigration, totalShards);
};
