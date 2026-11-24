"""SQLite-backed synthetic student and staff profile database."""

import json
import sqlite3
from pathlib import Path

from ai_support.agents.schema import CourseProgress, StaffProfile, StudentProfile
from ai_support.router.schema import Audience


class ProfileAuthorizationError(PermissionError):
    """Raised when a principal requests a profile outside its audience."""


class SyntheticProfileStore:
    def __init__(self, database_path: Path, seed_path: Path) -> None:
        self.database_path = database_path
        self.seed_path = seed_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS identities (
                    user_id TEXT PRIMARY KEY,
                    audience TEXT NOT NULL CHECK (audience IN ('student', 'staff'))
                );
                CREATE TABLE IF NOT EXISTS student_profiles (
                    user_id TEXT PRIMARY KEY REFERENCES identities(user_id),
                    name TEXT NOT NULL, programme TEXT NOT NULL, course TEXT NOT NULL,
                    enrolment_status TEXT NOT NULL, completed_subjects TEXT NOT NULL,
                    remaining_subjects TEXT NOT NULL, progress_percentage REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS staff_profiles (
                    user_id TEXT PRIMARY KEY REFERENCES identities(user_id),
                    name TEXT NOT NULL, job_title TEXT NOT NULL, department TEXT NOT NULL,
                    employment_status TEXT NOT NULL
                );
                """
            )
            count = connection.execute("SELECT COUNT(*) FROM identities").fetchone()[0]
            if count:
                return
            seed = json.loads(self.seed_path.read_text(encoding="utf-8"))
            for row in seed["students"]:
                connection.execute(
                    "INSERT INTO identities (user_id, audience) VALUES (?, 'student')",
                    (row["user_id"],),
                )
                connection.execute(
                    """INSERT INTO student_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        row["user_id"],
                        row["name"],
                        row["programme"],
                        row["course"],
                        row["enrolment_status"],
                        json.dumps(row["completed_subjects"]),
                        json.dumps(row["remaining_subjects"]),
                        row["progress_percentage"],
                    ),
                )
            for row in seed["staff"]:
                connection.execute(
                    "INSERT INTO identities (user_id, audience) VALUES (?, 'staff')",
                    (row["user_id"],),
                )
                connection.execute(
                    """INSERT INTO staff_profiles VALUES (?, ?, ?, ?, ?)""",
                    (
                        row["user_id"],
                        row["name"],
                        row["job_title"],
                        row["department"],
                        row["employment_status"],
                    ),
                )

    def audience_for_user(self, user_id: str) -> Audience | None:
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT audience FROM identities WHERE user_id = ?", (user_id,)
            ).fetchone()
        return row[0] if row else None

    def student_profile(self, authenticated_user_id: str) -> StudentProfile:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """SELECT user_id, name, programme, course, enrolment_status
                   FROM student_profiles WHERE user_id = ?""",
                (authenticated_user_id,),
            ).fetchone()
        if row is None:
            raise ProfileAuthorizationError("student profile is not available")
        return StudentProfile(**dict(row))

    def course_progress(self, authenticated_user_id: str) -> CourseProgress:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """SELECT completed_subjects, remaining_subjects, progress_percentage
                   FROM student_profiles WHERE user_id = ?""",
                (authenticated_user_id,),
            ).fetchone()
        if row is None:
            raise ProfileAuthorizationError("course progress is not available")
        return CourseProgress(
            completed_subjects=json.loads(row["completed_subjects"]),
            remaining_subjects=json.loads(row["remaining_subjects"]),
            progress_percentage=row["progress_percentage"],
        )

    def staff_profile(self, authenticated_user_id: str) -> StaffProfile:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """SELECT user_id, name, job_title, department, employment_status
                   FROM staff_profiles WHERE user_id = ?""",
                (authenticated_user_id,),
            ).fetchone()
        if row is None:
            raise ProfileAuthorizationError("staff profile is not available")
        return StaffProfile(**dict(row))
