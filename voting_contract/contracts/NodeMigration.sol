// SPDX-License-Identifier: GPL-3.0
pragma solidity 0.8.15;


contract NodeMigration {

    // 마이그레이션 정보 구조체
    struct MigrationInfo {
        address nodeAddress;         // 노드 주소
        uint256 fromShardId;         // 출발 샤드 ID
        uint256 toShardId;           // 도착 샤드 ID
        uint256 round;               // FL 라운드
        bytes32 dataIndexHash;       // 데이터 인덱스 정보의 해시 (train/test 인덱스)
        uint256 accuracy;            // 대상 샤드 모델에 대한 정확도 (소수점 2자리, 100배)
        uint256 timestamp;
        bool approved;
        bool executed;
    }

    // 샤드별 마이그레이션 결과 (Python의 p.shard_list에 해당)
    struct ShardMigrationResult {
        uint256 shardId;
        uint256 round;
        address[] incomingNodes;     // 들어오는 노드 목록
        address[] outgoingNodes;     // 나가는 노드 목록
        bytes32 migrationPayloadHash; // 마이그레이션 페이로드 해시 (train/test 인덱스)
        bool finalized;
    }

    // 정확도 평가 기록 (Python main.py의 acc_list에 해당)
    struct AccuracyRecord {
        address nodeAddress;
        uint256 round;
        uint256 shardId;             // 평가 대상 샤드
        uint256 accuracy;            // 정확도 (100배 정수)
    }

    address public owner;
    uint256 public totalMigrations;
    uint256 public totalShards;

    // 마이그레이션 ID => MigrationInfo
    mapping(uint256 => MigrationInfo) public migrations;
    // 라운드 => 샤드 ID => ShardMigrationResult
    mapping(uint256 => mapping(uint256 => ShardMigrationResult)) public shardMigrationResults;
    // 노드 주소 => 라운드 => 샤드 ID => AccuracyRecord
    mapping(address => mapping(uint256 => mapping(uint256 => AccuracyRecord))) public accuracyRecords;
    // 노드 주소 => 현재 샤드 ID
    mapping(address => uint256) public nodeCurrentShard;
    // 등록된 노드 여부
    mapping(address => bool) public registeredNodes;
    // 라운드 => 마이그레이션 ID 배열
    mapping(uint256 => uint256[]) public roundMigrations;

    // Events
    event AccuracySubmitted(address indexed nodeAddress, uint256 indexed round, uint256 indexed shardId, uint256 accuracy);
    event MigrationRequested(uint256 indexed migrationId, address indexed nodeAddress, uint256 fromShardId, uint256 toShardId, uint256 round);
    event MigrationApproved(uint256 indexed migrationId);
    event MigrationExecuted(uint256 indexed migrationId, address indexed nodeAddress, uint256 fromShardId, uint256 toShardId);
    event ShardMigrationFinalized(uint256 indexed round, uint256 indexed shardId, bytes32 payloadHash);
    event NodeRegistered(address indexed nodeAddress, uint256 indexed shardId);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function.");
        _;
    }

    modifier onlyRegisteredNode() {
        require(registeredNodes[msg.sender], "Node is not registered.");
        _;
    }

    constructor(uint256 _totalShards) {
        owner = msg.sender;
        totalMigrations = 0;
        totalShards = _totalShards;
    }

    function registerNode(address _nodeAddress, uint256 _shardId) external onlyOwner {
        require(!registeredNodes[_nodeAddress], "Node already registered.");
        require(_shardId > 0 && _shardId <= totalShards, "Invalid shard ID.");

        registeredNodes[_nodeAddress] = true;
        nodeCurrentShard[_nodeAddress] = _shardId;

        emit NodeRegistered(_nodeAddress, _shardId);
    }


    function submitAccuracy(uint256 _round, uint256 _shardId, uint256 _accuracy) external onlyRegisteredNode {
        require(_shardId > 0 && _shardId <= totalShards, "Invalid shard ID.");
        require(_accuracy <= 10000, "Accuracy cannot exceed 100%.");

        accuracyRecords[msg.sender][_round][_shardId] = AccuracyRecord({
            nodeAddress: msg.sender,
            round: _round,
            shardId: _shardId,
            accuracy: _accuracy
        });

        emit AccuracySubmitted(msg.sender, _round, _shardId, _accuracy);
    }

    function requestMigration(
        uint256 _round,
        uint256 _toShardId,
        bytes32 _dataIndexHash,
        uint256 _accuracy
    ) external onlyRegisteredNode {
        require(_toShardId > 0 && _toShardId <= totalShards, "Invalid shard ID.");

        uint256 fromShardId = nodeCurrentShard[msg.sender];

        totalMigrations++;
        migrations[totalMigrations] = MigrationInfo({
            nodeAddress: msg.sender,
            fromShardId: fromShardId,
            toShardId: _toShardId,
            round: _round,
            dataIndexHash: _dataIndexHash,
            accuracy: _accuracy,
            timestamp: block.timestamp,
            approved: false,
            executed: false
        });

        roundMigrations[_round].push(totalMigrations);

        // outgoing 기록
        if (fromShardId != _toShardId) {
            shardMigrationResults[_round][fromShardId].outgoingNodes.push(msg.sender);
            shardMigrationResults[_round][_toShardId].incomingNodes.push(msg.sender);
        }

        emit MigrationRequested(totalMigrations, msg.sender, fromShardId, _toShardId, _round);
    }


    function approveMigration(uint256 _migrationId) external onlyOwner {
        require(_migrationId > 0 && _migrationId <= totalMigrations, "Invalid migration ID.");
        MigrationInfo storage migration = migrations[_migrationId];
        require(!migration.approved, "Migration already approved.");

        migration.approved = true;

        emit MigrationApproved(_migrationId);
    }


    function executeMigration(uint256 _migrationId) external onlyOwner {
        MigrationInfo storage migration = migrations[_migrationId];
        require(migration.approved, "Migration not approved.");
        require(!migration.executed, "Migration already executed.");

        // 노드의 현재 샤드를 업데이트
        nodeCurrentShard[migration.nodeAddress] = migration.toShardId;
        migration.executed = true;

        emit MigrationExecuted(_migrationId, migration.nodeAddress, migration.fromShardId, migration.toShardId);
    }


    function finalizeShardMigration(uint256 _round, uint256 _shardId, bytes32 _payloadHash) external onlyOwner {
        require(_shardId > 0 && _shardId <= totalShards, "Invalid shard ID.");

        ShardMigrationResult storage result = shardMigrationResults[_round][_shardId];
        result.shardId = _shardId;
        result.round = _round;
        result.migrationPayloadHash = _payloadHash;
        result.finalized = true;

        emit ShardMigrationFinalized(_round, _shardId, _payloadHash);
    }


    function batchExecuteMigrations(uint256[] calldata _migrationIds) external onlyOwner {
        for (uint256 i = 0; i < _migrationIds.length; i++) {
            MigrationInfo storage migration = migrations[_migrationIds[i]];
            if (migration.approved && !migration.executed) {
                nodeCurrentShard[migration.nodeAddress] = migration.toShardId;
                migration.executed = true;
                emit MigrationExecuted(_migrationIds[i], migration.nodeAddress, migration.fromShardId, migration.toShardId);
            }
        }
    }


    function getRoundMigrations(uint256 _round) external view returns (uint256[] memory) {
        return roundMigrations[_round];
    }


    function getIncomingNodes(uint256 _round, uint256 _shardId) external view returns (address[] memory) {
        return shardMigrationResults[_round][_shardId].incomingNodes;
    }

    function getOutgoingNodes(uint256 _round, uint256 _shardId) external view returns (address[] memory) {
        return shardMigrationResults[_round][_shardId].outgoingNodes;
    }


    function getNodeShard(address _nodeAddress) external view returns (uint256) {
        require(registeredNodes[_nodeAddress], "Node not registered.");
        return nodeCurrentShard[_nodeAddress];
    }


    function getAccuracy(address _nodeAddress, uint256 _round, uint256 _shardId) external view returns (uint256) {
        return accuracyRecords[_nodeAddress][_round][_shardId].accuracy;
    }
}
