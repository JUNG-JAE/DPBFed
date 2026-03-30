// SPDX-License-Identifier: GPL-3.0
pragma solidity 0.8.15;

import "./ShardManager.sol";
import "./ModelRegistry.sol";
import "./GlobalAggregation.sol";
import "./NodeMigration.sol";


contract FederatedLearning {

    // FL 라운드 상태
    enum RoundState {
        INITIALIZED,          // 라운드 초기화됨
        SHARD_MODELS_UPLOADING,  // 샤드 모델 업로드 중
        VOTING_IN_PROGRESS,   // 투표 진행 중
        AGGREGATION_COMPLETE, // 집계 완료
        MIGRATION_PHASE,      // 마이그레이션 단계
        COMPLETED             // 라운드 완료
    }

    // 라운드 정보 구조체
    struct RoundInfo {
        uint256 roundNumber;
        RoundState state;
        uint256 uploadedShardCount;  // 업로드된 샤드 모델 수
        uint256 totalVotingSessions; // 총 투표 세션 수
        bytes32 finalGlobalModelHash; // 최종 글로벌 모델 해시
        uint256 startTime;
        uint256 endTime;
    }

    address public owner;

    // 서브 컨트랙트 주소
    ShardManager public shardManager;
    ModelRegistry public modelRegistry;
    GlobalAggregation public globalAggregation;
    NodeMigration public nodeMigration;

    uint256 public currentRound;
    uint256 public totalShards;

    // 라운드 => RoundInfo
    mapping(uint256 => RoundInfo) public rounds;

    // Events
    event RoundStarted(uint256 indexed round, uint256 timestamp);
    event ShardModelReceived(uint256 indexed round, uint256 indexed shardId, bytes32 modelHash);
    event AllShardModelsReceived(uint256 indexed round);
    event VotingPhaseStarted(uint256 indexed round);
    event AggregationCompleted(uint256 indexed round, bytes32 globalModelHash);
    event MigrationPhaseStarted(uint256 indexed round);
    event RoundCompleted(uint256 indexed round, uint256 timestamp);
    event SubContractsDeployed(address shardManager, address modelRegistry, address globalAggregation, address nodeMigration);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function.");
        _;
    }

    modifier inState(uint256 _round, RoundState _state) {
        require(rounds[_round].state == _state, "Invalid round state for this operation.");
        _;
    }

    constructor(uint256 _totalShards, uint256 _alphaNumerator, uint256 _alphaDenominator) {
        owner = msg.sender;
        totalShards = _totalShards;
        currentRound = 0;

        // 서브 컨트랙트 배포
        shardManager = new ShardManager(_totalShards);
        modelRegistry = new ModelRegistry();
        globalAggregation = new GlobalAggregation(_totalShards, _alphaNumerator, _alphaDenominator);
        nodeMigration = new NodeMigration(_totalShards);

        emit SubContractsDeployed(
            address(shardManager),
            address(modelRegistry),
            address(globalAggregation),
            address(nodeMigration)
        );
    }

    function startNewRound() external onlyOwner {
        if (currentRound > 0) {
            require(rounds[currentRound].state == RoundState.COMPLETED, "Current round is not completed.");
        }

        currentRound++;
        rounds[currentRound] = RoundInfo({
            roundNumber: currentRound,
            state: RoundState.INITIALIZED,
            uploadedShardCount: 0,
            totalVotingSessions: 0,
            finalGlobalModelHash: bytes32(0),
            startTime: block.timestamp,
            endTime: 0
        });

        emit RoundStarted(currentRound, block.timestamp);
    }

    function beginShardModelUpload() external onlyOwner inState(currentRound, RoundState.INITIALIZED) {
        rounds[currentRound].state = RoundState.SHARD_MODELS_UPLOADING;
    }

    function receiveShardModel(
        uint256 _shardId,
        bytes32 _modelHash,
        uint256 _cumulativeWeight
    ) external onlyOwner inState(currentRound, RoundState.SHARD_MODELS_UPLOADING) {
        require(_shardId > 0 && _shardId <= totalShards, "Invalid shard ID.");

        rounds[currentRound].uploadedShardCount++;

        emit ShardModelReceived(currentRound, _shardId, _modelHash);

        // 모든 샤드 모델이 업로드되었는지 확인
        // Python server_receiver.py의 len(shard_model_list) == p.SHARD_NUM 조건에 해당
        if (rounds[currentRound].uploadedShardCount >= totalShards) {
            emit AllShardModelsReceived(currentRound);
        }
    }

    function beginVotingPhase() external onlyOwner inState(currentRound, RoundState.SHARD_MODELS_UPLOADING) {
        require(rounds[currentRound].uploadedShardCount >= totalShards, "Not all shard models uploaded.");
        rounds[currentRound].state = RoundState.VOTING_IN_PROGRESS;

        emit VotingPhaseStarted(currentRound);
    }

    function completeAggregation(bytes32 _globalModelHash) external onlyOwner inState(currentRound, RoundState.VOTING_IN_PROGRESS) {
        require(_globalModelHash != bytes32(0), "Invalid global model hash.");

        rounds[currentRound].state = RoundState.AGGREGATION_COMPLETE;
        rounds[currentRound].finalGlobalModelHash = _globalModelHash;

        emit AggregationCompleted(currentRound, _globalModelHash);
    }

    function beginMigrationPhase() external onlyOwner inState(currentRound, RoundState.AGGREGATION_COMPLETE) {
        rounds[currentRound].state = RoundState.MIGRATION_PHASE;

        emit MigrationPhaseStarted(currentRound);
    }

    function completeRound() external onlyOwner inState(currentRound, RoundState.MIGRATION_PHASE) {
        rounds[currentRound].state = RoundState.COMPLETED;
        rounds[currentRound].endTime = block.timestamp;

        emit RoundCompleted(currentRound, block.timestamp);
    }

    function completeRoundOneAggregation(bytes32 _globalModelHash) external onlyOwner {
        require(currentRound == 1, "This function is only for round 1.");
        require(rounds[currentRound].state == RoundState.SHARD_MODELS_UPLOADING, "Invalid state.");
        require(rounds[currentRound].uploadedShardCount >= totalShards, "Not all shard models uploaded.");

        rounds[currentRound].state = RoundState.AGGREGATION_COMPLETE;
        rounds[currentRound].finalGlobalModelHash = _globalModelHash;

        emit AggregationCompleted(currentRound, _globalModelHash);
    }

    // ==================== View Functions ====================

    function getCurrentRoundInfo() external view returns (
        uint256 roundNumber,
        RoundState state,
        uint256 uploadedShardCount,
        uint256 totalVotingSessions,
        bytes32 finalGlobalModelHash,
        uint256 startTime,
        uint256 endTime
    ) {
        RoundInfo storage r = rounds[currentRound];
        return (r.roundNumber, r.state, r.uploadedShardCount, r.totalVotingSessions, r.finalGlobalModelHash, r.startTime, r.endTime);
    }

    function getSubContractAddresses() external view returns (
        address shardManagerAddr,
        address modelRegistryAddr,
        address globalAggregationAddr,
        address nodeMigrationAddr
    ) {
        return (
            address(shardManager),
            address(modelRegistry),
            address(globalAggregation),
            address(nodeMigration)
        );
    }

    function getRoundState(uint256 _round) external view returns (RoundState) {
        return rounds[_round].state;
    }

    function getRoundFinalModel(uint256 _round) external view returns (bytes32) {
        return rounds[_round].finalGlobalModelHash;
    }
}
