const Polling = artifacts.require("Polling");

module.exports = function (deployer) {
  // 예시 파라미터 (실제 배포 시 모델 해시와 타임스탬프를 설정)
  const model1Hash = "0x0000000000000000000000000000000000000000000000000000000000000001";
  const model1Timestamp = Math.floor(Date.now() / 1000);
  const model2Hash = "0x0000000000000000000000000000000000000000000000000000000000000002";
  const model2Timestamp = Math.floor(Date.now() / 1000);
  const preModelHash = "0x0000000000000000000000000000000000000000000000000000000000000000";
  const preModelTimestamp = Math.floor(Date.now() / 1000);

  deployer.deploy(Polling, model1Hash, model1Timestamp, model2Hash, model2Timestamp, preModelHash, preModelTimestamp);
};
