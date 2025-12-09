# 045.DSA_Capstone_Assignment_5
# Hospital Appointment & Triage System

## Overview
This Python project implements a **hospital appointment and triage system** with support for:

- Routine appointments
- Emergency triage
- Doctor schedule management
- Undo functionality
- Reporting and analytics

The system uses **queues, heaps, hash tables, and linked lists** to efficiently manage patients, doctors, and tokens.

---

## Features

1. **Patient Management**
   - Register or update patient details.
   - Track patient history (appointments served and pending).

2. **Doctor Management**
   - Add doctors with their specialization.
   - Maintain schedules (slots) for appointments.
   - Track pending and served patients.

3. **Appointment Booking**
   - Routine appointments via circular queues (FIFO).
   - Emergency triage via priority queue (min-heap by severity).

4. **Undo Functionality**
   - Undo the last action for slots, patient updates, bookings, or serving.

5. **Reports**
   - Doctor pending/served report.
   - Overall served vs pending report.
   - Top K frequent patients.

6. **CLI & Demo**
   - Command-line interface to interact with the system.
   - Demo function to showcase system functionalities.

---

## Data Structures Used

- **CircularQueue** → For routine appointment queue.
- **EmergencyQueue** (min-heap) → For emergency triage based on severity.
- **HashTable with Chaining** → For patient indexing and quick retrieval.
- **Linked List** → For doctor schedule slots.
- **UndoStack** → For undoing actions in the system.

---
hs = HospitalSystem()
hs.add_doctor(1, "Dr. Rao", "Cardiology", routine_capacity=5)
hs.add_doctor(2, "Dr. Meena", "General", routine_capacity=5)

p1 = Patient(1, "Amit", 30)
p2 = Patient(2, "Sunita", 25)
hs.patient_upsert(p1)
hs.patient_upsert(p2)

t1 = hs.book_routine(1, 1, slot_id=101)
t2 = hs.book_routine(2, 2, slot_id=201)
e1 = hs.triage_insert(2, severity_score=1, doctor_id=1)

served = hs.serve_next(1)
print("Served patient:", served)

```bash
git clone <repository_url>
