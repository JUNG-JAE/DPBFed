// SPDX-License-Identifier: GPL-3.0
pragma solidity 0.8.15;


contract ModelRegistry {

    // 로컬 모델 정보 구조체 (각 샤드에서 업로드하는 모델)
    struct LocalModel {
        bytes32 modelHash;          // 모델 파라미터의 해시값
        uint256 shardId;            // 모델이 속한 샤드 ID
        uint256 round;              // FL 라운드 번호
        address uploader;           // 모델을 업로드한 노드 주소
        uint256 timestamp;          // 업로드 시간
        uint256 cumulativeWeight;   // DAG에서의 누적 가중치 (CRS)
        bool verified;              // 검증 완료 여부
    }

    // 글로벌 모델 정보 구조체
    struct GlobalModel {
        bytes32 modelHash;          // 글로벌 모델의 해시값
        uint256 round;              // FL 라운드 번호
        uint256 aggregationStep;    // 글로벌 집계 단계 (ρ 값, 논문 Equation 9)
        uint256 timestamp;          // 생성 시간
        bool finalized;             // 최종 확정 여부
    }

    address public owner;
    uint256 public currentRound;
    uint256 public totalLocalModels;
    uint256 public totalGlobalModels;

    // 로컬 모델 ID => LocalModel
    mapping(uint256 => LocalModel) public localModels;
    // 글로벌 모델 ID => GlobalModel
    mapping(uint256 => GlobalModel) public globalModels;
    // 라운드 => 샤드 ID => 로컬 모델 ID 배열
    mapping(uint256 => mapping(uint256 => uint256[])) public roundShardModels;
    // 라운드 => 글로벌 모델 ID 배열
    mapping(uint256 => uint256[]) public roundGlobalModels;
    // 라운드 => 업로드된 샤드 수
    mapping(uint256 => uint256) public roundUploadedShardCount;
    // 라운드 => 최종 글로벌 모델 해시
    mapping(uint256 => bytes32) public roundFinalGlobalModel;

    // 등록된 샤드 주소 (업로드 권한)
    mapping(address => bool) public authorizedUploaders;

    // Events
    event LocalModelUploaded(uint256 indexed modelId, uint256 indexed shardId, uint256 indexed round, bytes32 modelHash, address uploader);
    event GlobalModelCreated(uint256 indexed modelId, uint256 indexed round, uint256 aggregationStep, bytes32 modelHash);
    event GlobalModelFinalized(uint256 indexed round, bytes32 finalModelHash);
    event RoundAdvanced(uint256 indexed newRound);
    event UploaderAuthorized(address indexed uploader);
    event UploaderRevoked(address indexed uploader);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function.");
        _;
    }

    modifier onlyAuthorizedUploader() {
        require(authorizedUploaders[msg.sender] || msg.sender == owner, "Not authorized to upload models.");
        _;
    }

    constructor() {
        owner = msg.sender;
        currentRound = 1;
        totalLocalModels = 0;
        totalGlobalModels = 0;
    }

    /**
     * @dev 업로더 권한을 부여한다
     */
    function authorizeUploader(address _uploader) external onlyOwner {
        authorizedUploaders[_uploader] = true;
        emit UploaderAuthorized(_uploader);
    }

    /**
     * @dev 업로더 권한을 해제한다
     */
    function revokeUploader(address _uploader) external onlyOwner {
        authorizedUploaders[_uploader] = false;
        emit UploaderRevoked(_uploader);
    }


    function uploadLocalModel(
        bytes32 _modelHash,
        uint256 _shardId,
        uint256 _round,
        uint256 _cumulativeWeight
    ) external onlyAuthorizedUploader {
        require(_round == currentRound, "Model round does not match current round.");
        require(_modelHash != bytes32(0), "Invalid model hash.");

        totalLocalModels++;
        localModels[totalLocalModels] = LocalModel({
            modelHash: _modelHash,
            shardId: _shardId,
            round: _round,
            uploader: msg.sender,
            timestamp: block.timestamp,
            cumulativeWeight: _cumulativeWeight,
            verified: false
        });

        roundShardModels[_round][_shardId].push(totalLocalModels);
        roundUploadedShardCount[_round]++;

        emit LocalModelUploaded(totalLocalModels, _shardId, _round, _modelHash, msg.sender);
    }


    function verifyLocalModel(uint256 _modelId) external onlyOwner {
        require(_modelId > 0 && _modelId <= totalLocalModels, "Invalid model ID.");
        localModels[_modelId].verified = true;
    }


    function registerGlobalModel(
        bytes32 _modelHash,
        uint256 _aggregationStep
    ) external onlyOwner {
        require(_modelHash != bytes32(0), "Invalid model hash.");

        totalGlobalModels++;
        globalModels[totalGlobalModels] = GlobalModel({
            modelHash: _modelHash,
            round: currentRound,
            aggregationStep: _aggregationStep,
            timestamp: block.timestamp,
            finalized: false
        });

        roundGlobalModels[currentRound].push(totalGlobalModels);

        emit GlobalModelCreated(totalGlobalModels, currentRound, _aggregationStep, _modelHash);
    }


    function finalizeGlobalModel(uint256 _globalModelId) external onlyOwner {
        require(_globalModelId > 0 && _globalModelId <= totalGlobalModels, "Invalid global model ID.");
        GlobalModel storage gm = globalModels[_globalModelId];
        require(gm.round == currentRound, "Model is not from the current round.");

        gm.finalized = true;
        roundFinalGlobalModel[currentRound] = gm.modelHash;

        emit GlobalModelFinalized(currentRound, gm.modelHash);
    }


    function advanceRound() external onlyOwner {
        require(roundFinalGlobalModel[currentRound] != bytes32(0), "Current round global model is not finalized.");
        currentRound++;
        emit RoundAdvanced(currentRound);
    }


    function getShardModels(uint256 _round, uint256 _shardId) external view returns (uint256[] memory) {
        return roundShardModels[_round][_shardId];
    }


    function getRoundGlobalModels(uint256 _round) external view returns (uint256[] memory) {
        return roundGlobalModels[_round];
    }


    function getFinalGlobalModel(uint256 _round) external view returns (bytes32) {
        return roundFinalGlobalModel[_round];
    }


    function getLocalModelInfo(uint256 _modelId) external view returns (
        bytes32 modelHash,
        uint256 shardId,
        uint256 round,
        address uploader,
        uint256 timestamp,
        uint256 cumulativeWeight,
        bool verified
    ) {
        LocalModel storage lm = localModels[_modelId];
        return (lm.modelHash, lm.shardId, lm.round, lm.uploader, lm.timestamp, lm.cumulativeWeight, lm.verified);
    }
}
