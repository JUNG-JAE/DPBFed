// SPDX-License-Identifier: GPL-3.0
pragma solidity 0.8.15;

contract Polling {

    // Voter 구조체
    struct Voter {
        uint weight; // 투표자의 가중치
        bool voted;  // 재투표 방지를 위한 변수
        address permission; // 투표 권한을 확인하기 위한 변수
        uint vote;   // index of the voted proposal
    }

    // Model 구조체
    struct Model {
        bytes32 model_hash;   // short name (up to 32 bytes)
        uint timestamp; // model create time
        uint vote_count; // number of accumulated votes
    }

    struct ModelInfo {
        bytes32 model_hash;
        uint timestamp;
    }

    address public owner;

    // Voter 구조체를 voters list에 저장
    mapping(address => Voter) public voters;

    Model[] public models;
    ModelInfo[] public model_info;

    event modelInformation(bytes32 model_hash, uint timestamp);

    // smart contract를 deploy할때 model1, model1_timestamp, model2, model2_timestamp 입력
    constructor(bytes32 _model1, uint _model1_timestamp, bytes32 _model2, uint _model2_timestamp, bytes32 _pre_model, uint _pre_model_timestamp) {
        owner = msg.sender;

        models.push(Model({
            model_hash:_model1,
            timestamp:_model1_timestamp,
            vote_count: 0
        }));

        models.push(Model({
            model_hash:_model2,
            timestamp:_model2_timestamp,
            vote_count: 0
        }));

        model_info.push(ModelInfo({
            model_hash:_pre_model,
            timestamp:_pre_model_timestamp
        }));

        model_info.push(ModelInfo({
            model_hash:_model1,
            timestamp:_model1_timestamp
        }));

        model_info.push(ModelInfo({
            model_hash:_model2,
            timestamp:_model2_timestamp
        }));

    }

    // voting 권한 부여
    function grantPermission(address voter) external {
        require(msg.sender == owner, "Only the owner can grant permissions.");
        require(!voters[voter].voted, "The voter already voted.");
        require(voters[voter].weight == 0);
        
        // main node 는 투표를 할 수 없게 하기 위해
        if (msg.sender != voter){
            voters[voter].weight = 1;
        }
    }

    /// Give your vote (including votes delegated to you) to model_index `proposals[model_index].model_hash`.
    function vote(uint model_index) external {
        Voter storage sender = voters[msg.sender];
        require(sender.weight != 0, "Has no permission to vote");
        require(!sender.voted, "Already voted.");
        sender.voted = true;
        sender.vote = model_index;

        // If `model_index` is out of the range of the array, this will throw automatically and revert all changes.
        models[model_index].vote_count += sender.weight;
    }

    /// @dev Computes the winning proposal taking all previous votes into account.
    function electedModelIndexer() public view
            returns (uint electedModel_)
    {
        uint maximumVoteCount = 0;
        for (uint i = 0; i < models.length; i++) {
            if (models[i].vote_count > maximumVoteCount) {
                maximumVoteCount = models[i].vote_count;
                electedModel_ = i;
            }
        }
    }

    // Calls winningProposal() function to get the index of the winner contained in the proposals array and then returns the model_hash of the winner
    function end() external view
            returns (bytes32 electedModel_)
    {
        electedModel_ = models[electedModelIndexer()].model_hash;
    }

    function getModelInfo() external
    {
        for(uint i = 0; i < model_info.length; i++){
            emit modelInformation(model_info[i].model_hash, model_info[i].timestamp);
        }
    }

}

