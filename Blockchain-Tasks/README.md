# Blockchain Audition Tasks (2025)

Welcome to the audition process! We're looking for curious and creative minds. The following tasks are designed to test your ability to grasp new concepts and think critically. No prior experience is required, but you will be asked to think logically and write some simple code. Let's begin!

## Easy Task: Explore a Decentralized World

**Objective:** Understand how blockchain enables verifiable digital ownership by exploring a real-world Non-Fungible Token (NFT).

### Steps:

1. Go to a major NFT marketplace like [OpenSea](https://opensea.io). You do not need to connect a wallet or buy anything. You will only be exploring.

2. Browse and find any NFT that you find interesting (it could be art, a collectible, a gaming item, etc.).

3. Click on the NFT to view its details page. Look for sections like "Properties," "Details" (which includes the Contract Address), and "Item Activity."

4. In a short paragraph, describe the NFT you chose. In a second paragraph, explain what the "Item Activity" section tells you. How is seeing the complete history of an item different from how we track ownership of physical items?

### Docs and Resources:

- [OpenSea Marketplace](https://opensea.io) (for exploring)
- [What is an NFT? - Forbes](https://www.forbes.com/advisor/investing/nft-non-fungible-token/)

### Video Tutorial:

- [NFTs, Simply Explained](https://www.youtube.com/watch?v=NNQLJcJEzv0) (YouTube - 8 minutes)

---

## Medium Task: Design and Code the Logic for a Trust Machine

**Objective:** Conceptually design a private blockchain solution and write pseudocode for its core logic.

### Steps:

**The Scenario:** Imagine your university needs a secure digital voting system for student elections. The goals are to prevent fraud, maintain privacy, and ensure auditable results.

#### Part 1 (Design):

On a single page, briefly outline the system. Answer these questions:

- **Participants:** Who would interact with this blockchain? (e.g., University Admin, Registered Students).
- **Permissions:** What special permissions would each participant have? (e.g., Who can create the ballot? Who is allowed to vote?).

#### Part 2 (A Little Coding):

- **Data Structure:** Show what the data for a single ballot might look like using JSON format. Include fields like `electionID`, `candidateName`, and `studentVoterID`.
- **Logic:** Write pseudocode for a function called `isVoteValid()`. This function should take a `studentVoterID` as input and check two things: 1. if the student is on the official list of eligible voters, and 2. if that student has already voted. It should return `true` or `false`. _(Pseudocode is simplified code written in plain English, focusing on logic, not perfect syntax)_.

### Docs and Resources:

- [What is JSON? - A Simple Introduction](https://www.json.org/json-en.html)
- [An Introduction to Pseudocode](https://www.khanacademy.org/computing/computer-programming/programming/intro-to-programming/a/pseudocode)

### Video Tutorial:

- [How Blockchain Can Revolutionize Elections](https://www.youtube.com/watch?v=BT9h5sfhCW4) (YouTube - 4 minutes)
- [Programming Logic: How To Think Like A Programmer](https://www.youtube.com/watch?v=azcrPFhaY9k) (YouTube - 11 minutes)

---

## Hard Task: Code Your First Mini-Blockchain

**Objective:** Understand the fundamental "chain" structure of a blockchain by writing a simple Python program.

### Steps:

1. Go to the online Python editor [Replit](https://replit.com) and create a new Python project. You do not need to install anything on your computer.

2. **The Goal:** Write a simple Python script to create a "blockchain" with three blocks. Each block must contain its own hash and the hash of the previous block, creating a chain.

### Your Coding Task:

1. You will need a function to calculate a hash. Use Python's built-in `hashlib` library. The hash for a block should be created from its `index`, `timestamp`, `data`, and `previous_hash`.

2. Create a "Genesis Block" (the very first block) with a `previous_hash` of `"0"`.

3. Create a second block. Its `previous_hash` must be the hash of the Genesis Block.

4. Create a third block. Its `previous_hash` must be the hash of the second block.

5. Print out your final chain of blocks to the console, showing how they are linked.

6. Submit the link to your public Replit project.

### Docs and Resources:

- [Go to Replit to start coding](https://replit.com)
- [A Beginner's Guide to Hashing in Python (hashlib)](https://docs.python.org/3/library/hashlib.html)
- [Starter Code Snippet/Guide](https://github.com/dvf/blockchain) (for guidance)

### Video Tutorial:

- [Code Your Own Blockchain in 20 minutes with Python](https://www.youtube.com/watch?v=_160oMzblY8) (YouTube - 20 minutes)
- [Python Hashing with hashlib](https://www.youtube.com/watch?v=3K4Pyy6d-zk) (YouTube - 7 minutes)
