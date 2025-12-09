import heapq
import time
from typing import Optional, List, Any, Callable, Dict, Tuple

class Patient:
    def __init__(self, patient_id: int, name: str, age: int, severity: int = 0):
        self.id = patient_id
        self.name = name
        self.age = age
        self.severity = severity
        self.history: List[Any] = [] 
    def __repr__(self) -> str:
        return f"Patient(id={self.id}, name='{self.name}', age={self.age}, severity={self.severity})"


class Token:
    ROUTINE = "ROUTINE"
    EMERGENCY = "EMERGENCY"

    def __init__(self, token_id: int, patient_id: int, doctor_id: int, slot_id: Optional[int], typ: str):
        self.token_id = token_id
        self.patient_id = patient_id
        self.doctor_id = doctor_id
        self.slot_id = slot_id
        self.type = typ
        self.timestamp = time.time()

    def __repr__(self) -> str:
        return (f"Token(id={self.token_id}, p={self.patient_id}, d={self.doctor_id}, "
                f"slot={self.slot_id}, type={self.type})")


class SlotNode:
    def __init__(self, slot_id: int, start_time: str, end_time: str, status: str = "AVAILABLE"):
        self.slot_id = slot_id
        self.start_time = start_time
        self.end_time = end_time
        self.status = status  
        self.next: Optional['SlotNode'] = None

    def __repr__(self) -> str:
        return f"Slot(slot_id={self.slot_id}, {self.start_time}-{self.end_time}, status={self.status})"

# Circular Queue (Ring Buffer)

class CircularQueue:
    def __init__(self, capacity: int = 100):
        if capacity < 1:
            raise ValueError("Capacity must be >= 1")
        self.capacity = capacity
        self.buffer: List[Optional[Any]] = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def enqueue(self, item: Any) -> bool:
        """Return True if enqueued, False if queue full."""
        if self.size == self.capacity:
            return False
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1
        return True

    def dequeue(self) -> Optional[Any]:
        """Return item or None if empty."""
        if self.size == 0:
            return None
        item = self.buffer[self.head]
        self.buffer[self.head] = None
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return item

    def peek(self) -> Optional[Any]:
        if self.size == 0:
            return None
        return self.buffer[self.head]

    def is_empty(self) -> bool:
        return self.size == 0

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"CircularQueue(cap={self.capacity}, size={self.size})"

    def to_list(self) -> List[Any]:
        """Return a list of elements in queue order (read-only)."""
        out = []
        i = self.head
        for _ in range(self.size):
            out.append(self.buffer[i])
            i = (i + 1) % self.capacity
        return out
    
# Emergency Queue (Min-Heap)

class EmergencyQueue:
    """
    Min-heap where lower severity_score => higher priority.
    We store tuples (severity, counter, token)
    counter breaks ties to ensure stable ordering.
    """

    def __init__(self):
        self.heap: List[Tuple[int, int, Token]] = []
        self.counter = 0

    def insert(self, severity_score: int, token: Token) -> None:
        heapq.heappush(self.heap, (severity_score, self.counter, token))
        self.counter += 1

    def pop(self) -> Optional[Token]:
        if not self.heap:
            return None
        _, _, token = heapq.heappop(self.heap)
        return token

    def peek(self) -> Optional[Token]:
        if not self.heap:
            return None
        return self.heap[0][2]

    def __len__(self) -> int:
        return len(self.heap)

    def __repr__(self) -> str:
        return f"EmergencyQueue(size={len(self.heap)})"

    def rebuild_excluding(self, token_id: int) -> None:
        """Remove an arbitrary token by rebuilding heap excluding token_id."""
        arr: List[Tuple[int, int, Token]] = []
        while self.heap:
            s, c, t = heapq.heappop(self.heap)
            if t.token_id != token_id:
                arr.append((s, c, t))
        for item in arr:
            heapq.heappush(self.heap, item)

# Hash Table with Chaining (Patient Index)

class HashTable:
    def __init__(self, bucket_count: int = 211):
        self.bucket_count = bucket_count
        self.buckets: List[List[Patient]] = [[] for _ in range(bucket_count)]
        self.size = 0

    def _hash(self, key: int) -> int:
        return key % self.bucket_count

    def insert(self, patient: Patient) -> None:
        idx = self._hash(patient.id)
        bucket = self.buckets[idx]
        for i, p in enumerate(bucket):
            if p.id == patient.id:
                bucket[i] = patient
                return
        bucket.append(patient)
        self.size += 1

    def get(self, patient_id: int) -> Optional[Patient]:
        idx = self._hash(patient_id)
        for p in self.buckets[idx]:
            if p.id == patient_id:
                return p
        return None

    def delete(self, patient_id: int) -> bool:
        idx = self._hash(patient_id)
        bucket = self.buckets[idx]
        for i, p in enumerate(bucket):
            if p.id == patient_id:
                bucket.pop(i)
                self.size -= 1
                return True
        return False

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"HashTable(buckets={self.bucket_count}, size={self.size})"

# Undo Stack

class UndoAction:
    def __init__(self, action_type: str, payload: Any, revert_fn: Callable[[], None]):
        self.action_type = action_type
        self.payload = payload
        self.revert_fn = revert_fn
        self.timestamp = time.time()

    def revert(self) -> None:
        self.revert_fn()

    def __repr__(self) -> str:
        return f"UndoAction(type={self.action_type}, payload={self.payload})"


class UndoStack:
    def __init__(self):
        self._stack: List[UndoAction] = []

    def push(self, action: UndoAction) -> None:
        self._stack.append(action)

    def pop(self) -> Optional[UndoAction]:
        if not self._stack:
            return None
        return self._stack.pop()

    def peek(self) -> Optional[UndoAction]:
        if not self._stack:
            return None
        return self._stack[-1]

    def __len__(self) -> int:
        return len(self._stack)

class Doctor:
    def __init__(self, doctor_id: int, name: str, specialization: str, routine_capacity: int = 50):
        self.id = doctor_id
        self.name = name
        self.specialization = specialization
        self.schedule_head: Optional[SlotNode] = None
        self.routine_queue = CircularQueue(capacity=routine_capacity)
        self.served_count = 0
        self.pending_count = 0

    def __repr__(self) -> str:
        return f"Doctor(id={self.id}, name='{self.name}', spec='{self.specialization}')"

    def add_slot_head(self, slot: SlotNode) -> None:
        slot.next = self.schedule_head
        self.schedule_head = slot

    def find_slot(self, slot_id: int) -> Optional[SlotNode]:
        cur = self.schedule_head
        while cur:
            if cur.slot_id == slot_id:
                return cur
            cur = cur.next
        return None

    def traverse_slots(self) -> List[SlotNode]:
        res: List[SlotNode] = []
        cur = self.schedule_head
        while cur:
            res.append(cur)
            cur = cur.next
        return res
    
# Hospital System (Core Orchestration)

class HospitalSystem:
    def __init__(self):
        self.doctors: Dict[int, Doctor] = {}
        self.patients = HashTable(bucket_count=211)
        self.emergency_queue = EmergencyQueue()
        self.undo_stack = UndoStack()
        self.next_token_id = 1
        self.served_tokens: List[Token] = []
        self.token_index: Dict[int, Token] = {}

    def add_doctor(self, doctor_id: int, name: str, specialization: str, routine_capacity: int = 50) -> None:
        if doctor_id in self.doctors:
            raise ValueError(f"Doctor {doctor_id} already exists.")
        self.doctors[doctor_id] = Doctor(doctor_id, name, specialization, routine_capacity)

    def schedule_add_slot(self, doctor_id: int, slot_id: int, start_time: str, end_time: str) -> None:
        doctor = self.doctors.get(doctor_id)
        if not doctor:
            raise ValueError("Doctor not found")
        slot = SlotNode(slot_id, start_time, end_time, status="AVAILABLE")
        doctor.add_slot_head(slot)

        def revert_remove():
            self._remove_slot_node(doctor, slot_id)
        self.undo_stack.push(UndoAction("add_slot", (doctor_id, slot_id), revert_remove))

    def _remove_slot_node(self, doctor: Doctor, slot_id: int) -> bool:
        prev = None
        cur = doctor.schedule_head
        while cur:
            if cur.slot_id == slot_id:
                if prev:
                    prev.next = cur.next
                else:
                    doctor.schedule_head = cur.next
                return True
            prev = cur
            cur = cur.next
        return False

    def schedule_cancel(self, doctor_id: int, slot_id: int) -> bool:
        doctor = self.doctors.get(doctor_id)
        if not doctor:
            return False
        slot = doctor.find_slot(slot_id)
        if not slot:
            return False
        old_status = slot.status
        slot.status = "CANCELLED"

        def revert():
            slot.status = old_status

        self.undo_stack.push(UndoAction("cancel_slot", (doctor_id, slot_id), revert))
        return True

    def schedule_traverse(self, doctor_id: int) -> List[SlotNode]:
        doctor = self.doctors.get(doctor_id)
        if not doctor:
            return []
        return doctor.traverse_slots()

    def patient_upsert(self, patient: Patient) -> None:
        prev = self.patients.get(patient.id)
        self.patients.insert(patient)

        def revert():
            if prev:
                self.patients.insert(prev)
            else:
                self.patients.delete(patient.id)

        self.undo_stack.push(UndoAction("patient_upsert", patient.id, revert))

    def patient_get(self, patient_id: int) -> Optional[Patient]:
        return self.patients.get(patient_id)

    def patient_delete(self, patient_id: int) -> bool:
        patient = self.patients.get(patient_id)
        if not patient:
            return False
        self.patients.delete(patient_id)

        def revert():
            self.patients.insert(patient)

        self.undo_stack.push(UndoAction("patient_delete", patient_id, revert))
        return True

    def book_routine(self, patient_id: int, doctor_id: int, slot_id: Optional[int] = None) -> Optional[Token]:
        patient = self.patient_get(patient_id)
        doctor = self.doctors.get(doctor_id)
        if not patient or not doctor:
            return None

        token = Token(self.next_token_id, patient_id, doctor_id, slot_id, Token.ROUTINE)
        success = doctor.routine_queue.enqueue(token)
        if not success:
            return None

        self.token_index[token.token_id] = token
        self.next_token_id += 1
        doctor.pending_count += 1
        patient.history.append(token.token_id)

        def revert():
            self._remove_token_from_doctor_queue(doctor, token.token_id)
            self.token_index.pop(token.token_id, None)
            doctor.pending_count = max(0, doctor.pending_count - 1)
            if token.token_id in patient.history:
                patient.history.remove(token.token_id)

        self.undo_stack.push(UndoAction("book_routine", token.token_id, revert))
        return token

    def _remove_token_from_doctor_queue(self, doctor: Doctor, token_id: int) -> None:
        """Rebuild queue excluding the token_id."""
        tmp: List[Token] = []
        while not doctor.routine_queue.is_empty():
            t = doctor.routine_queue.dequeue()
            if t and t.token_id != token_id:
                tmp.append(t)
        for t in tmp:
            doctor.routine_queue.enqueue(t)

    def serve_next(self, doctor_id: int) -> Optional[Token]:
        doctor = self.doctors.get(doctor_id)
        if not doctor:
            return None

        temp: List[Token] = []
        served: Optional[Token] = None
        while len(self.emergency_queue) > 0:
            cand = self.emergency_queue.pop()
            if cand.doctor_id == doctor_id:
                served = cand
                break
            temp.append(cand)

        for t in temp:
         
            pat = self.patient_get(t.patient_id)
            sev = pat.severity if pat else 100
            self.emergency_queue.insert(sev, t)

        if served:
            served.type = Token.EMERGENCY
            self._mark_served(doctor, served)
            return served

        token = doctor.routine_queue.dequeue()
        if token:
            self._mark_served(doctor, token)
            return token

        return None

    def _mark_served(self, doctor: Doctor, token: Token) -> None:
        doctor.served_count += 1
        doctor.pending_count = max(0, doctor.pending_count - 1)
        self.served_tokens.append(token)
        self.token_index.pop(token.token_id, None)

        patient = self.patient_get(token.patient_id)
        if patient:
            patient.history.append(f"served:{token.token_id}")

        def revert():
        
            if token.type == Token.EMERGENCY:
                pat = self.patient_get(token.patient_id)
                sev = pat.severity if pat else 100
                self.emergency_queue.insert(sev, token)
            else:
        
                tmp: List[Token] = []
                while not doctor.routine_queue.is_empty():
                    tmp.append(doctor.routine_queue.dequeue())
                doctor.routine_queue.enqueue(token)
                for t in tmp:
                    doctor.routine_queue.enqueue(t)
                doctor.pending_count += 1

            doctor.served_count = max(0, doctor.served_count - 1)
            try:
                self.served_tokens.remove(token)
            except ValueError:
                pass

        self.undo_stack.push(UndoAction("serve", token.token_id, revert))

    def triage_insert(self, patient_id: int, severity_score: int, doctor_id: int) -> Optional[Token]:
        patient = self.patient_get(patient_id)
        doctor = self.doctors.get(doctor_id)
        if not patient or not doctor:
            return None

        token = Token(self.next_token_id, patient_id, doctor_id, None, Token.EMERGENCY)
        self.next_token_id += 1
        patient.severity = severity_score
        self.emergency_queue.insert(severity_score, token)
        self.token_index[token.token_id] = token
        doctor.pending_count += 1
        patient.history.append(token.token_id)

        def revert():
            self.emergency_queue.rebuild_excluding(token.token_id)
            self.token_index.pop(token.token_id, None)
            doctor.pending_count = max(0, doctor.pending_count - 1)
            if token.token_id in patient.history:
                patient.history.remove(token.token_id)

        self.undo_stack.push(UndoAction("triage_insert", token.token_id, revert))
        return token

    def undo_pop(self) -> bool:
        action = self.undo_stack.pop()
        if not action:
            return False
        action.revert()
        return True

    def report_doctor_pending(self, doctor_id: int) -> Dict[str, Any]:
        doctor = self.doctors.get(doctor_id)
        if not doctor:
            raise ValueError("Doctor not found")
        next_slot = None
        cur = doctor.schedule_head
        while cur:
            if cur.status == "AVAILABLE":
                next_slot = cur
                break
            cur = cur.next
        return {
            "doctor_id": doctor_id,
            "doctor_name": doctor.name,
            "pending_count": doctor.pending_count,
            "served_count": doctor.served_count,
            "next_available_slot": next_slot
        }

    def report_served_vs_pending(self) -> Dict[str, int]:
        total_pending = sum(d.pending_count for d in self.doctors.values())
        total_served = sum(d.served_count for d in self.doctors.values())
        return {"served": total_served, "pending": total_pending}

    def top_k_frequent_patients(self, k: int = 5) -> List[Tuple[int, Patient]]:
        freq: List[Tuple[int, Patient]] = []
        for bucket in self.patients.buckets:
            for p in bucket:
                freq.append((len(p.history), p))
        freq.sort(reverse=True, key=lambda x: x[0])
        return freq[:k]

    def debug_state(self) -> None:
        print("=== Hospital State ===")
        print("Doctors:")
        for doc in self.doctors.values():
            print(f"  {doc} | pending={doc.pending_count} | served={doc.served_count} | queue_size={len(doc.routine_queue)}")
            print("   schedule:", doc.traverse_slots())
            print("   queue:", doc.routine_queue.to_list())
        print("Patients count:", len(self.patients))
        print("Emergency queue size:", len(self.emergency_queue))
        print("Undo stack size:", len(self.undo_stack))
        print("Served tokens:", self.served_tokens)
        print("======================")

# Demo & Simple CLI

def demo() -> None:
    """Small demo that exercises primary flows."""
    hs = HospitalSystem()

    hs.add_doctor(1, "Dr. Rao", "Cardiology", routine_capacity=5)
    hs.add_doctor(2, "Dr. Meena", "General", routine_capacity=5)

    hs.schedule_add_slot(1, 101, "09:00", "09:15")
    hs.schedule_add_slot(1, 102, "09:15", "09:30")
    hs.schedule_add_slot(2, 201, "10:00", "10:15")

    p1 = Patient(1, "Amit", 30)
    p2 = Patient(2, "Sunita", 25)
    p3 = Patient(3, "Rahul", 40)
    hs.patient_upsert(p1)
    hs.patient_upsert(p2)
    hs.patient_upsert(p3)

    t1 = hs.book_routine(1, 1, slot_id=101)
    t2 = hs.book_routine(2, 1, slot_id=102)
    t3 = hs.book_routine(3, 2, slot_id=201)
    print("Booked tokens:", t1, t2, t3)

    e1 = hs.triage_insert(3, severity_score=2, doctor_id=1)
    print("Inserted emergency:", e1)

    served1 = hs.serve_next(1)
    print("Served (doctor 1):", served1)

    served2 = hs.serve_next(1)
    print("Served (doctor 1):", served2)

    undone = hs.undo_pop()
    print("Undo last action result:", undone)

    print("Doctor 1 report:", hs.report_doctor_pending(1))
    print("Overall served vs pending:", hs.report_served_vs_pending())
    print("Top patients:", hs.top_k_frequent_patients(3))

    hs.debug_state()


def run_cli() -> None:
    hs = HospitalSystem()
    print("Welcome to Hospital Appointment & Triage System (CLI demo)\n")

    hs.add_doctor(1, "Dr. Rao", "Cardiology", routine_capacity=10)
    hs.add_doctor(2, "Dr. Meena", "General", routine_capacity=10)

    def safe_int(prompt: str) -> int:
        while True:
            try:
                return int(input(prompt))
            except ValueError:
                print("Enter a valid integer.")

    while True:
        print("\nMenu:")
        print("1) Register/Upsert Patient")
        print("2) Add Doctor Slot")
        print("3) Book Routine Appointment")
        print("4) Emergency Intake (Triage)")
        print("5) Serve Next Patient (by Doctor)")
        print("6) Undo Last Action")
        print("7) Reports")
        print("8) Debug State")
        print("9) Run Demo")
        print("0) Exit")
        choice = input("Select option: ").strip()
        if choice == "1":
            pid = safe_int("Patient ID: ")
            name = input("Name: ").strip()
            age = safe_int("Age: ")
            p = Patient(pid, name, age)
            hs.patient_upsert(p)
            print(f"Patient {pid} upserted.")
        elif choice == "2":
            did = safe_int("Doctor ID: ")
            sid = safe_int("Slot ID: ")
            st = input("Start time (e.g. 09:00): ").strip()
            et = input("End time (e.g. 09:15): ").strip()
            try:
                hs.schedule_add_slot(did, sid, st, et)
                print("Slot added.")
            except ValueError as e:
                print("Error:", e)
        elif choice == "3":
            pid = safe_int("Patient ID: ")
            did = safe_int("Doctor ID: ")
            sid_in = input("Slot ID (optional, press enter to skip): ").strip()
            sid = int(sid_in) if sid_in else None
            tok = hs.book_routine(pid, did, sid)
            if tok:
                print("Booked:", tok)
            else:
                print("Booking failed (missing patient/doctor or queue full).")
        elif choice == "4":
            pid = safe_int("Patient ID: ")
            did = safe_int("Doctor ID: ")
            sev = safe_int("Severity (lower = more urgent): ")
            tok = hs.triage_insert(pid, sev, did)
            if tok:
                print("Emergency token:", tok)
            else:
                print("Triage failed (missing patient/doctor).")
        elif choice == "5":
            did = safe_int("Doctor ID: ")
            tok = hs.serve_next(did)
            if tok:
                print("Served:", tok)
            else:
                print("Nothing to serve for this doctor.")
        elif choice == "6":
            ok = hs.undo_pop()
            print("Undo performed." if ok else "Nothing to undo.")
        elif choice == "7":
            print("Doctors:")
            for doc in hs.doctors.values():
                print(f"  {doc.id}: {doc.name} ({doc.specialization}) - pending={doc.pending_count}, served={doc.served_count}")
            print("Overall:", hs.report_served_vs_pending())
        elif choice == "8":
            hs.debug_state()
        elif choice == "9":
            demo()
        elif choice == "0":
            print("Exiting. Bye.")
            break
        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    run_cli()
    
