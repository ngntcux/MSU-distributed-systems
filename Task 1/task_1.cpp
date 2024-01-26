#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

namespace {
constexpr int ROOT_ID = 0;
constexpr int NUM_PROCESSES = 16;
constexpr int DEFAULT_TAG = 0;
}  // namespace

class Graph {
   public:
    Graph() {
        outgoingAdjacencyList_[0] = {};
        outgoingAdjacencyList_[1] = {0};
        outgoingAdjacencyList_[2] = {6};
        outgoingAdjacencyList_[3] = {2};
        outgoingAdjacencyList_[4] = {0};
        outgoingAdjacencyList_[5] = {1};
        outgoingAdjacencyList_[6] = {5};
        outgoingAdjacencyList_[7] = {3};
        outgoingAdjacencyList_[8] = {4};
        outgoingAdjacencyList_[9] = {13};
        outgoingAdjacencyList_[10] = {9};
        outgoingAdjacencyList_[11] = {7};
        outgoingAdjacencyList_[12] = {8};
        outgoingAdjacencyList_[13] = {12};
        outgoingAdjacencyList_[14] = {10};
        outgoingAdjacencyList_[15] = {11};

        incomingAdjacencyList_[0] = {1, 4};
        incomingAdjacencyList_[1] = {5};
        incomingAdjacencyList_[2] = {3};
        incomingAdjacencyList_[3] = {7};
        incomingAdjacencyList_[4] = {8};
        incomingAdjacencyList_[5] = {6};
        incomingAdjacencyList_[6] = {2};
        incomingAdjacencyList_[7] = {11};
        incomingAdjacencyList_[8] = {12};
        incomingAdjacencyList_[9] = {10};
        incomingAdjacencyList_[10] = {14};
        incomingAdjacencyList_[11] = {15};
        incomingAdjacencyList_[12] = {13};
        incomingAdjacencyList_[13] = {9};
        incomingAdjacencyList_[14] = {};
        incomingAdjacencyList_[15] = {};
    }
    const unordered_set<int>& getNext(int id) const { return outgoingAdjacencyList_.at(id); }
    const unordered_set<int>& getPrev(int id) const { return incomingAdjacencyList_.at(id); }
    int getSize() const { return max(outgoingAdjacencyList_.size(), incomingAdjacencyList_.size()); }

    int getRemainingMessages(int id) const {
        int count = 0;
        while (!incomingAdjacencyList_.at(id).empty()) {
            id = *incomingAdjacencyList_.at(id).begin();
            ++count;
        }
        return count;
    }
    int getIdFromPositionBackward(int postionId, int index) const {
        int id = postionId;
        int total = this->getRemainingMessages(postionId);
        for (int i = 0; i < min(index, total); ++i) {
            id = *incomingAdjacencyList_.at(id).begin();
        }
        return id;
    }

   private:
    unordered_map<int, unordered_set<int>> outgoingAdjacencyList_;
    unordered_map<int, unordered_set<int>> incomingAdjacencyList_;
};

int getRandomInt() {
    static random_device randomDevice;
    static mt19937 randomNumberGenerator(randomDevice());
    static constexpr int lowerBound = 1;
    static constexpr int upperBound = 100;
    static uniform_int_distribution<int> distribution(lowerBound, upperBound);
    return distribution(randomNumberGenerator);
}

int main() {
    MPI_Init(NULL, NULL);
    int numProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    int processId;
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    if (numProcesses != NUM_PROCESSES) {
        if (processId == ROOT_ID) {
            cerr << "Error: The program supports execution only on " << NUM_PROCESSES << " MPI processes." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    Graph graph;

    int data = getRandomInt();

    cout << "My id: " << processId << ". My data: " << data << "." << endl;

    vector<int> recvData;

    if (processId == ROOT_ID) {
        recvData.resize(graph.getSize());

        recvData[ROOT_ID] = data;

        vector<MPI_Request> requests(numProcesses - 1);
        int count = 0;
        for (int i = 0; i < numProcesses; ++i) {
            if (i != ROOT_ID) {
                MPI_Irecv(&recvData[i], 1, MPI_INT, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, &requests[count]);
                ++count;
            }
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);

        for (auto elem : recvData) {
            cout << elem << " ";
        }
        cout << endl;
    } else if (graph.getNext(processId).find(ROOT_ID) != graph.getNext(processId).end()) {
        MPI_Send(&data, 1, MPI_INT, *graph.getNext(processId).begin(), processId, MPI_COMM_WORLD);

        int countMessages = graph.getRemainingMessages(processId);
        for (int i = 0; i < countMessages; ++i) {
            MPI_Recv(&data, 1, MPI_INT, *graph.getPrev(processId).begin(), MPI_ANY_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Send(&data, 1, MPI_INT, *graph.getNext(processId).begin(),
                     graph.getIdFromPositionBackward(processId, i + 1), MPI_COMM_WORLD);
        }
    } else {
        MPI_Send(&data, 1, MPI_INT, *graph.getNext(processId).begin(), DEFAULT_TAG, MPI_COMM_WORLD);

        int countMessages = graph.getRemainingMessages(processId);
        for (int i = 0; i < countMessages; ++i) {
            MPI_Recv(&data, 1, MPI_INT, *graph.getPrev(processId).begin(), MPI_ANY_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Send(&data, 1, MPI_INT, *graph.getNext(processId).begin(), DEFAULT_TAG, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
