# DAG-Based Blockchain Sharding for Secure Federated Learning with Non-IID Data

## Abstract
Federated learning allows multiple clients to train a shared model without exposing raw data, which makes it promising for privacy-sensitive and distributed environments. In practice, however, its reliability is limited by two major issues: strongly non-IID local data and vulnerability to malicious updates injected by adversarial participants. This becomes even more challenging when the system cannot depend on a fully trusted central aggregator. To address these problems, this repository provides an implementation of a hierarchical blockchain-based federated learning framework. The framework is designed to support asynchronous training settings while improving robustness against poisoning attacks under heterogeneous data distributions.

**[Access the research paper](https://www.mdpi.com/1424-8220/22/21/8263)**

## Key Contributions
- Introduces a hierarchical two-layer aggregation design that separates shard-level model integration from global aggregation, reducing dependence on a single trusted coordinator.
- Proposes a DAG-based selection mechanism that considers local performance, model similarity, and structural diversity to make malicious updates less influential.
- Demonstrates stable learning behavior under severe non-IID conditions and poisoning attacks, including scenarios with a high fraction of malicious participants.

## Overview Architecture

![Overview Architecture](https://github.com/user-attachments/assets/00fb7aa0-2877-42b0-b961-a2aae8380864)

This framework consists of two layers: a DAG-based shard layer and a global layer built on the main blockchain.

## Performance under Poisoning Attacks

| Model Poisoning | Data Poisoning | Label-Swapping |
|---|---|---|
| ![Model Poisoning Accuracy](https://github.com/user-attachments/assets/d8b90c7b-7c32-485f-85af-0f98ee621bd5) | ![Data Poisoning Accuracy](https://github.com/user-attachments/assets/9e24dbde-eea3-499e-8409-73aaa81d9b74) | ![Label Swapping Accuracy](https://github.com/user-attachments/assets/452533ff-879b-4118-9465-ef3423f07330) |
| **Accuracy** | **Accuracy** | **Accuracy** |
| ![Model Poisoning RSR](https://github.com/user-attachments/assets/dfc883fd-4c09-4b01-b79b-8c8ef289f221) | ![Data Poisoning RSR](https://github.com/user-attachments/assets/d08a6dcd-a758-423c-a4ec-aa0e9ab66767) | ![Label Swapping RSR](https://github.com/user-attachments/assets/16e3fd39-1fe7-4676-bb41-aca2fd09a044) |
| **RSR** | **RSR** | **RSR** |

### Attack Settings
- **Model poisoning attack**: an attacker directly perturbs or manipulates model parameters before sharing the local update, with the goal of corrupting the aggregation process.
- **Data poisoning attack**: an attacker trains on intentionally corrupted data so that the resulting model update indirectly degrades the global model.
- **Label-swapping attack**: an attacker changes the labels of local training samples, producing misleading gradients while still using otherwise normal-looking data.

### What is RSR?
**Reference Score Ratio (RSR)** indicates how much influence malicious updates gain during the selection process. In the paper, it is defined as the ratio of the cumulative reference score assigned to malicious-node transactions relative to the total reference score in each round. A lower RSR means poisoned models are rarely selected or referenced by other nodes, whereas a higher RSR suggests that malicious updates are receiving more attention inside the DAG-based shard layer.

## Code Explanation

### Contents
The uploaded code is a simplified version of the experimental simulator used in the paper. Because of GitHub storage limitations, the full experimental datasets are not included in this repository. In the original experiment, the system used five shards, but this public version has been reduced to two shards for easier reproduction. In addition, a model server simulator is provided for testing before deploying the actual Ethereum-based voting contract.

- **model_server**: integrates shard-level models uploaded from the shard layer and provides a voting function. This is an experimental simulator used before the actual smart-contract-based deployment.
- **shard**: implements the DAG-based shard structure and local shard behavior.
- **voting_contract**: contains the code for Ethereum-based voting.

## How to Run

```bash
cd DAG_Blockchain_FL
pip3 install -r requirement

# Option 1
.example_bash.sh

# Option 2 (requires 3 terminals)

# Terminal 1
cd model_server
python3 handler.py

# Terminal 2
cd shard1
python3 main.py

# Terminal 3
cd shard2
python3 main.py
```
