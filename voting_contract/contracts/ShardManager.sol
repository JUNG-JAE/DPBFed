// SPDX-License-Identifier: GPL-3.0
pragma solidity 0.8.15;


contract ShardManager {

    // 샤드 정보 구조체
    struct Shard {
        uint256 shardId;
        bool active;
        uint256 nodeCount;
        uint256 createdAt;
    }

    // 노드 정보 구조체
    struct Node {
        address nodeAddress;
        uint256 currentShardId;
        bool registered;
        uint256 registeredAt;
    }

    // 마이그레이션 요청 구조체
    struct MigrationRequest {
        address nodeAddress;
        uint256 fromShardId;
        uint256 toShardId;
        uint256 requestedAt;
        bool executed;
    }

    address public owner;
    uint256 public totalShards;
    uint256 public totalNodes;
    uint256 public migrationRequestCount;

    // 샤드 ID => Shard 구조체
    mapping(uint256 => Shard) public shards;
    // 노드 주소 => Node 구조체
    mapping(address => Node) public nodes;
    // 샤드 ID => 노드 주소 배열
    mapping(uint256 => address[]) public shardNodes;
    // 마이그레이션 요청 ID => MigrationRequest
    mapping(uint256 => MigrationRequest) public migrationRequests;

    // Events
    event ShardRegistered(uint256 indexed shardId, uint256 timestamp);
    event NodeRegistered(address indexed nodeAddress, uint256 indexed shardId, uint256 timestamp);
    event NodeMigrated(address indexed nodeAddress, uint256 indexed fromShardId, uint256 indexed toShardId, uint256 timestamp);
    event MigrationRequested(uint256 indexed requestId, address indexed nodeAddress, uint256 fromShardId, uint256 toShardId);
    event ShardDeactivated(uint256 indexed shardId, uint256 timestamp);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function.");
        _;
    }

    modifier onlyRegisteredNode() {
        require(nodes[msg.sender].registered, "Node is not registered.");
        _;
    }

    modifier validShard(uint256 _shardId) {
        require(_shardId > 0 && _shardId <= totalShards, "Invalid shard ID.");
        require(shards[_shardId].active, "Shard is not active.");
        _;
    }

    constructor(uint256 _initialShardCount) {
        owner = msg.sender;
        totalShards = 0;
        totalNodes = 0;
        migrationRequestCount = 0;

        // 초기 샤드 생성 (논문에서 |S|개의 샤드를 초기화)
        for (uint256 i = 0; i < _initialShardCount; i++) {
            _createShard();
        }
    }


    function _createShard() internal {
        totalShards++;
        shards[totalShards] = Shard({
            shardId: totalShards,
            active: true,
            nodeCount: 0,
            createdAt: block.timestamp
        });
        emit ShardRegistered(totalShards, block.timestamp);
    }


    function registerShard() external onlyOwner {
        _createShard();
    }


    function registerNode(uint256 _shardId) external validShard(_shardId) {
        require(!nodes[msg.sender].registered, "Node is already registered.");

        nodes[msg.sender] = Node({
            nodeAddress: msg.sender,
            currentShardId: _shardId,
            registered: true,
            registeredAt: block.timestamp
        });

        shardNodes[_shardId].push(msg.sender);
        shards[_shardId].nodeCount++;
        totalNodes++;

        emit NodeRegistered(msg.sender, _shardId, block.timestamp);
    }


    function requestMigration(uint256 _toShardId) external onlyRegisteredNode validShard(_toShardId) {
        uint256 currentShardId = nodes[msg.sender].currentShardId;
        require(currentShardId != _toShardId, "Already in the target shard.");

        migrationRequestCount++;
        migrationRequests[migrationRequestCount] = MigrationRequest({
            nodeAddress: msg.sender,
            fromShardId: currentShardId,
            toShardId: _toShardId,
            requestedAt: block.timestamp,
            executed: false
        });

        emit MigrationRequested(migrationRequestCount, msg.sender, currentShardId, _toShardId);
    }


    function executeMigration(uint256 _requestId) external onlyOwner {
        MigrationRequest storage request = migrationRequests[_requestId];
        require(!request.executed, "Migration already executed.");
        require(request.nodeAddress != address(0), "Invalid migration request.");
        require(shards[request.toShardId].active, "Target shard is not active.");

        address nodeAddr = request.nodeAddress;
        uint256 fromShardId = request.fromShardId;
        uint256 toShardId = request.toShardId;

        // 이전 샤드에서 노드를 제거
        _removeNodeFromShard(nodeAddr, fromShardId);

        // 새 샤드에 노드를 추가
        shardNodes[toShardId].push(nodeAddr);
        shards[toShardId].nodeCount++;

        // 노드 정보 업데이트
        nodes[nodeAddr].currentShardId = toShardId;

        // 마이그레이션 요청 완료 처리
        request.executed = true;

        emit NodeMigrated(nodeAddr, fromShardId, toShardId, block.timestamp);
    }


    function _removeNodeFromShard(address _nodeAddr, uint256 _shardId) internal {
        address[] storage nodeList = shardNodes[_shardId];
        for (uint256 i = 0; i < nodeList.length; i++) {
            if (nodeList[i] == _nodeAddr) {
                nodeList[i] = nodeList[nodeList.length - 1];
                nodeList.pop();
                break;
            }
        }
        shards[_shardId].nodeCount--;
    }


    function deactivateShard(uint256 _shardId) external onlyOwner validShard(_shardId) {
        shards[_shardId].active = false;
        emit ShardDeactivated(_shardId, block.timestamp);
    }


    function getShardNodes(uint256 _shardId) external view returns (address[] memory) {
        return shardNodes[_shardId];
    }

  
    function getShardNodeCount(uint256 _shardId) external view returns (uint256) {
        return shards[_shardId].nodeCount;
    }


    function getNodeShard(address _nodeAddr) external view returns (uint256) {
        require(nodes[_nodeAddr].registered, "Node is not registered.");
        return nodes[_nodeAddr].currentShardId;
    }


    function getActiveShardCount() external view returns (uint256 count) {
        for (uint256 i = 1; i <= totalShards; i++) {
            if (shards[i].active) {
                count++;
            }
        }
    }
}
