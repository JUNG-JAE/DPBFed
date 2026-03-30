// SPDX-License-Identifier: GPL-3.0
pragma solidity 0.8.15;


contract GlobalAggregation {

    // 투표 세션 구조체
    struct VotingSession {
        uint256 sessionId;
        uint256 round;                // FL 라운드
        uint256 votingOrder;          // 투표 순서 (2부터 시작, mitigate_update.py의 self.voting_order)
        bytes32[] candidateModels;    // 후보 모델 해시 배열
        string[] candidateNames;      // 후보 모델 이름 배열
        uint256[] voteCounts;         // 각 후보의 득표 수
        uint256 totalVotes;           // 총 투표 수
        uint256 electedIndex;         // 선출된 모델의 인덱스
        bool finalized;               // 투표 완료 여부
        uint256 createdAt;
    }

    // 투표 참가자 (커미티 멤버) 구조체
    struct CommitteeMember {
        address memberAddress;
        uint256 shardId;
        uint256 weight;              // 투표 가중치 (ex_voting.sol의 Voter.weight에 해당)
        bool hasVoted;
    }

    // Mitigate Update 파라미터 (Python의 GLOBAL_MODEL_ALPHA)
    struct MitigateParams {
        uint256 alphaNumerator;      // alpha의 분자 (정수 비율로 표현)
        uint256 alphaDenominator;    // alpha의 분모
    }

    address public owner;
    uint256 public totalSessions;
    uint256 public currentRound;
    uint256 public totalShards;

    MitigateParams public mitigateParams;

    // 세션 ID => VotingSession
    mapping(uint256 => VotingSession) public votingSessions;
    // 세션 ID => 커미티 멤버 주소 => CommitteeMember
    mapping(uint256 => mapping(address => CommitteeMember)) public sessionMembers;
    // 세션 ID => 커미티 멤버 주소 배열
    mapping(uint256 => address[]) public sessionMemberList;
    // 라운드 => 투표 커미티 샤드 ID 배열 (Python의 self.voting_committee)
    mapping(uint256 => uint256[]) public roundVotingCommittee;
    // 라운드 => 현재 글로벌 모델 해시 (g1, g2, g3 등의 중간 결과)
    mapping(uint256 => mapping(uint256 => bytes32)) public roundIntermediateModels;

    // Events
    event VotingSessionCreated(uint256 indexed sessionId, uint256 indexed round, uint256 votingOrder, uint256 candidateCount);
    event CommitteeMemberAdded(uint256 indexed sessionId, address indexed member, uint256 shardId);
    event VoteCast(uint256 indexed sessionId, address indexed voter, uint256 candidateIndex);
    event VotingFinalized(uint256 indexed sessionId, uint256 electedIndex, bytes32 electedModelHash);
    event IntermediateModelSaved(uint256 indexed round, uint256 votingOrder, bytes32 modelHash);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function.");
        _;
    }

    constructor(uint256 _totalShards, uint256 _alphaNumerator, uint256 _alphaDenominator) {
        owner = msg.sender;
        totalSessions = 0;
        currentRound = 1;
        totalShards = _totalShards;

        // mitigate alpha = _alphaNumerator / _alphaDenominator (기본값: 1/2 = 0.5)
        mitigateParams = MitigateParams({
            alphaNumerator: _alphaNumerator,
            alphaDenominator: _alphaDenominator
        });
    }

    function createFirstVotingSession(
        uint256 _round,
        uint256 _shardA,
        uint256 _shardB,
        bytes32[] calldata _candidateHashes,
        string[] calldata _candidateNames
    ) external onlyOwner {
        require(_candidateHashes.length == 4, "First voting requires exactly 4 candidates.");
        require(_candidateHashes.length == _candidateNames.length, "Hash and name arrays must match.");

        totalSessions++;

        VotingSession storage session = votingSessions[totalSessions];
        session.sessionId = totalSessions;
        session.round = _round;
        session.votingOrder = 2;
        session.totalVotes = 0;
        session.electedIndex = 0;
        session.finalized = false;
        session.createdAt = block.timestamp;

        for (uint256 i = 0; i < _candidateHashes.length; i++) {
            session.candidateModels.push(_candidateHashes[i]);
            session.candidateNames.push(_candidateNames[i]);
            session.voteCounts.push(0);
        }

        // 투표 커미티에 샤드 추가
        roundVotingCommittee[_round].push(_shardA);
        roundVotingCommittee[_round].push(_shardB);

        emit VotingSessionCreated(totalSessions, _round, 2, 4);
    }

    function createSubsequentVotingSession(
        uint256 _round,
        uint256 _votingOrder,
        uint256 _newShardId,
        bytes32[] calldata _candidateHashes,
        string[] calldata _candidateNames
    ) external onlyOwner {
        require(_candidateHashes.length == 2, "Subsequent voting requires exactly 2 candidates.");
        require(_candidateHashes.length == _candidateNames.length, "Hash and name arrays must match.");
        require(_votingOrder > 2, "Voting order must be greater than 2 for subsequent sessions.");

        totalSessions++;

        VotingSession storage session = votingSessions[totalSessions];
        session.sessionId = totalSessions;
        session.round = _round;
        session.votingOrder = _votingOrder;
        session.totalVotes = 0;
        session.electedIndex = 0;
        session.finalized = false;
        session.createdAt = block.timestamp;

        for (uint256 i = 0; i < _candidateHashes.length; i++) {
            session.candidateModels.push(_candidateHashes[i]);
            session.candidateNames.push(_candidateNames[i]);
            session.voteCounts.push(0);
        }

        // 투표 커미티에 새 샤드 추가
        roundVotingCommittee[_round].push(_newShardId);

        emit VotingSessionCreated(totalSessions, _round, _votingOrder, 2);
    }

    function addCommitteeMember(uint256 _sessionId, address _member, uint256 _shardId) external onlyOwner {
        require(_sessionId > 0 && _sessionId <= totalSessions, "Invalid session ID.");
        require(!votingSessions[_sessionId].finalized, "Session already finalized.");
        require(!sessionMembers[_sessionId][_member].hasVoted, "Member already exists.");
        require(_member != owner, "Owner cannot be a committee member.");

        sessionMembers[_sessionId][_member] = CommitteeMember({
            memberAddress: _member,
            shardId: _shardId,
            weight: 1,
            hasVoted: false
        });

        sessionMemberList[_sessionId].push(_member);

        emit CommitteeMemberAdded(_sessionId, _member, _shardId);
    }

    function castVote(uint256 _sessionId, uint256 _candidateIndex) external {
        VotingSession storage session = votingSessions[_sessionId];
        require(!session.finalized, "Voting session is already finalized.");
        require(_candidateIndex < session.candidateModels.length, "Invalid candidate index.");

        CommitteeMember storage member = sessionMembers[_sessionId][msg.sender];
        require(member.weight > 0, "Not a committee member or no voting weight.");
        require(!member.hasVoted, "Already voted.");

        member.hasVoted = true;
        session.voteCounts[_candidateIndex] += member.weight;
        session.totalVotes++;

        emit VoteCast(_sessionId, msg.sender, _candidateIndex);
    }

    function finalizeVoting(uint256 _sessionId) external onlyOwner {
        VotingSession storage session = votingSessions[_sessionId];
        require(!session.finalized, "Session already finalized.");

        uint256 maxVotes = 0;
        uint256 electedIdx = 0;

        for (uint256 i = 0; i < session.voteCounts.length; i++) {
            if (session.voteCounts[i] > maxVotes) {
                maxVotes = session.voteCounts[i];
                electedIdx = i;
            }
        }

        session.electedIndex = electedIdx;
        session.finalized = true;

        // 중간 글로벌 모델 저장
        roundIntermediateModels[session.round][session.votingOrder] = session.candidateModels[electedIdx];

        emit VotingFinalized(_sessionId, electedIdx, session.candidateModels[electedIdx]);
        emit IntermediateModelSaved(session.round, session.votingOrder, session.candidateModels[electedIdx]);
    }


    function updateMitigateParams(uint256 _alphaNumerator, uint256 _alphaDenominator) external onlyOwner {
        require(_alphaDenominator > 0, "Denominator cannot be zero.");
        mitigateParams.alphaNumerator = _alphaNumerator;
        mitigateParams.alphaDenominator = _alphaDenominator;
    }


    function setRound(uint256 _round) external onlyOwner {
        currentRound = _round;
    }

    function getSessionCandidates(uint256 _sessionId) external view returns (bytes32[] memory) {
        return votingSessions[_sessionId].candidateModels;
    }


    function getSessionCandidateNames(uint256 _sessionId) external view returns (string[] memory) {
        return votingSessions[_sessionId].candidateNames;
    }


    function getSessionVoteCounts(uint256 _sessionId) external view returns (uint256[] memory) {
        return votingSessions[_sessionId].voteCounts;
    }


    function getRoundVotingCommittee(uint256 _round) external view returns (uint256[] memory) {
        return roundVotingCommittee[_round];
    }


    function getElectedModelHash(uint256 _sessionId) external view returns (bytes32) {
        require(votingSessions[_sessionId].finalized, "Voting not finalized yet.");
        return votingSessions[_sessionId].candidateModels[votingSessions[_sessionId].electedIndex];
    }


    function getIntermediateModel(uint256 _round, uint256 _votingOrder) external view returns (bytes32) {
        return roundIntermediateModels[_round][_votingOrder];
    }


    function getSessionMembers(uint256 _sessionId) external view returns (address[] memory) {
        return sessionMemberList[_sessionId];
    }
}
