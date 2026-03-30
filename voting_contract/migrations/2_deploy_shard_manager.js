const ShardManager = artifacts.require("ShardManager");

module.exports = function (deployer) {
  // 초기 샤드 수: 2 (parameter.py의 SHARD_NUM = 2)
  const initialShardCount = 2;
  deployer.deploy(ShardManager, initialShardCount);
};
