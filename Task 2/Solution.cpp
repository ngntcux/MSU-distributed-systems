#include <mpi-ext.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <thread>

using namespace std;

namespace {
constexpr int REQUEST = 0;
constexpr int REPLY = 1;
constexpr int MSG_LEN = 3;
constexpr int DEFAULT_TAG = 0;
constexpr int MIN_NUM_PROCESSES = 2;
static const string CHECKPOINT = "Checkpoint.dat";
}  // namespace

class LogicalClock {
   private:
    int timestamp_;

   public:
    LogicalClock() { timestamp_ = 0; }
    LogicalClock(int timestamp) : timestamp_(timestamp) {}
    int getTimestamp() const { return timestamp_; }
    void setTimestamp(int timestamp) { timestamp_ = timestamp; }
    void tick() { ++timestamp_; }
    void update(int timestamp) { timestamp_ = max(timestamp_, timestamp) + 1; }
};

class Event {
   private:
    int id_, type_, timestamp_;

   public:
    Event() {
        id_ = 0;
        type_ = 0;
        timestamp_ = 0;
    }
    Event(int id, int type, int timestamp) : id_(id), type_(type), timestamp_(timestamp) {}

    int getId() const { return id_; }
    int getType() const { return type_; }
    int getTimestamp() const { return timestamp_; }
    void setId(int id) { id_ = id; }
    void setType(int type) { type_ = type; }
    void setTimestamp(int timestamp) { timestamp_ = timestamp; }

    bool operator<(const Event& other) const {
        return timestamp_ < other.getTimestamp() || (timestamp_ == other.getTimestamp() && id_ < other.getId());
    }
};

void saveCheckpoint(MPI_Comm comm, const string& filename, const LogicalClock& clock, const Event& event) {
    int processId;
    MPI_Comm_rank(comm, &processId);
    MPI_File file;
    MPI_File_open(comm, filename.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);

    const int timestamp = clock.getTimestamp();
    MPI_Offset offset = processId * sizeof(int) * (1 + MSG_LEN);
    MPI_File_write_at(file, offset, &timestamp, 1, MPI_INT, MPI_STATUS_IGNORE);

    offset += sizeof(int);
    const array<int, MSG_LEN> msg = {event.getId(), event.getType(), event.getTimestamp()};
    MPI_File_write_at(file, offset, &msg, msg.size(), MPI_INT, MPI_STATUS_IGNORE);

    MPI_File_close(&file);
}

void loadCheckpoint(MPI_Comm comm, const string& filename, LogicalClock& clock, Event& event) {
    int processId;
    MPI_Comm_rank(comm, &processId);
    MPI_File file;
    MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

    MPI_Offset offset = processId * sizeof(int) * (1 + MSG_LEN);
    int timestamp;
    MPI_File_read_at(file, offset, &timestamp, 1, MPI_INT, MPI_STATUS_IGNORE);
    clock.setTimestamp(timestamp);

    offset += sizeof(int);
    array<int, MSG_LEN> msg;
    MPI_File_read_at(file, offset, &msg, MSG_LEN, MPI_INT, MPI_STATUS_IGNORE);
    event.setId(msg[0]);
    event.setType(msg[1]);
    event.setTimestamp(msg[2]);

    MPI_File_close(&file);
}

void broadcast(MPI_Comm comm, const LogicalClock& clock, const Event& event) {
    int numProcesses;
    MPI_Comm_size(comm, &numProcesses);
    for (int nextId = 0; nextId < numProcesses; ++nextId) {
        if (nextId != event.getId()) {
            const array<int, MSG_LEN> buffer = {event.getId(), event.getType(), clock.getTimestamp()};
            MPI_Send(buffer.data(), buffer.size(), MPI_INT, nextId, DEFAULT_TAG, comm);
        }
    }
}

void send(MPI_Comm comm, const LogicalClock& clock, const Event& event, int toId) {
    const array<int, MSG_LEN> buffer = {event.getId(), event.getType(), clock.getTimestamp()};
    MPI_Send(buffer.data(), buffer.size(), MPI_INT, toId, DEFAULT_TAG, comm);
}

Event receive(MPI_Comm comm) {
    array<int, MSG_LEN> buffer{};
    MPI_Recv(buffer.data(), buffer.size(), MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
    return Event(buffer[0], buffer[1], buffer[2]);
}

chrono::milliseconds generateRandomDelay() {
    static random_device randomDevice;
    static mt19937 randomNumberGenerator(randomDevice());
    static constexpr int lowerBound = 1000;
    static constexpr int upperBound = 2000;
    static uniform_int_distribution<int> distribution(lowerBound, upperBound);
    return chrono::milliseconds(distribution(randomNumberGenerator));
}

int getRandomInt(int lowerBound = 0, int upperBound = 100) {
    static random_device randomDevice;
    static mt19937 randomNumberGenerator(randomDevice());
    uniform_int_distribution<int> distribution(lowerBound, upperBound);
    return distribution(randomNumberGenerator);
}

void criticalSection(MPI_Comm comm) {
    int processId;
    MPI_Comm_rank(comm, &processId);

    const filesystem::path filePath = "critical.txt";

    if (filesystem::exists(filePath)) {
        cerr << "Error (Process " << processId << "): File '" + filePath.string() + "' already exists!" << endl;
        MPI_Abort(comm, EXIT_FAILURE);
    }

    ofstream file(filePath);
    if (!file.is_open()) {
        cerr << "Error (Process " << processId << "): Unable to open file '" + filePath.string() + "'!" << endl;
        MPI_Abort(comm, EXIT_FAILURE);
    }
    file.close();

    auto delay = generateRandomDelay();
    cout << "Process " << processId << " sleeping for " << delay.count() << " milliseconds." << endl;
    this_thread::sleep_for(delay);

    if (!filesystem::remove(filePath)) {
        cerr << "Error (Process " << processId << "): Unable to remove file '" + filePath.string() + "'!" << endl;
        MPI_Abort(comm, EXIT_FAILURE);
    }

    cout << "Process " << processId << " exiting critical section." << endl;
}

bool ricartAgrawalaAlgorithm(MPI_Comm comm, LogicalClock& clock, Event& event) {
    int processId;
    MPI_Comm_rank(comm, &processId);
    int numProcesses;
    MPI_Comm_size(comm, &numProcesses);

    if (event.getId() == processId) {
        broadcast(comm, clock, event);

        set<Event> deferred;
        int numReplies = 0;
        while (numReplies < numProcesses - 1) {
            auto receivedEvent = receive(comm);
            if (receivedEvent.getType() == REQUEST) {
                if (receivedEvent.getTimestamp() < clock.getTimestamp() ||
                    (receivedEvent.getTimestamp() == clock.getTimestamp() && receivedEvent.getId() < event.getId())) {
                    Event tempEvent(processId, REPLY, clock.getTimestamp());
                    send(comm, clock, tempEvent, receivedEvent.getId());
                } else {
                    deferred.insert(receivedEvent);
                }
            }
            if (receivedEvent.getType() == REPLY) {
                ++numReplies;
            }
        }

        criticalSection(comm);

        for (const auto& nextEvent : deferred) {
            Event tempEvent(processId, REPLY, clock.getTimestamp());
            send(comm, clock, tempEvent, nextEvent.getId());
        }
    }

    return false;
}

void errorHandler(MPI_Comm* comm, int* errorCode, ...) {
    // int error = *errorCode;
    // char errorString[MPI_MAX_ERROR_STRING];
    // int communicatorSize, failedGroupSize, stringLength;

    // MPI_Group failedGroup;
    // MPI_Comm_size(*comm, &communicatorSize);

    // MPIX_Comm_failure_ack(*comm);
    // MPIX_Comm_failure_get_acked(*comm, &failedGroup);

    // MPI_Group_size(failedGroup, &failedGroupSize);
    // MPI_Error_string(error, errorString, &stringLength);

    // MPI_Comm newComm;
    // MPIX_Comm_shrink(*comm, &newComm);

    // int processId;
    // MPI_Comm_rank(newComm, &processId);

    // *comm = newComm;
}

int main() {
    MPI_Init(NULL, NULL);

    int globalProcessId;
    MPI_Comm_rank(MPI_COMM_WORLD, &globalProcessId);
    int globalNumProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &globalNumProcesses);
    if (globalNumProcesses < MIN_NUM_PROCESSES) {
        cerr << "Error: The program requires a minimum of " << MIN_NUM_PROCESSES << " MPI processes for execution."
             << endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Comm comm;
    int color = globalProcessId < globalNumProcesses - 1 ? 1 : 0;
    MPI_Comm_split(MPI_COMM_WORLD, color, globalProcessId, &comm);

    LogicalClock clock;
    Event currentEvent;

    if (color == 1) {
        MPI_Errhandler commErrhandler;
        MPI_Comm_create_errhandler(errorHandler, &commErrhandler);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, commErrhandler);

        int processId;
        MPI_Comm_rank(comm, &processId);
        int numProcesses;
        MPI_Comm_size(comm, &numProcesses);

        clock.setTimestamp(getRandomInt());
        currentEvent.setId(processId);
        currentEvent.setType(REQUEST);
        currentEvent.setTimestamp(clock.getTimestamp());

        saveCheckpoint(comm, CHECKPOINT, clock, currentEvent);
        if (ricartAgrawalaAlgorithm(comm, clock, currentEvent)) {
            loadCheckpoint(comm, CHECKPOINT, clock, currentEvent);
            ricartAgrawalaAlgorithm(comm, clock, currentEvent);
        }
    }

    MPI_Comm_free(&comm);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
