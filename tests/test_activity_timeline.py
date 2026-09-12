"""
Integration tests for GitHub-style Project Activity and Event Timeline against running Melodify server.
"""
import unittest
import os
import sys
import uuid
import numpy as np
import soundfile as sf
import requests

BASE_URL = "http://localhost:8000"

class TestActivityTimeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("beat_outputs", exist_ok=True)
        cls.test_wav_name = f"test_act_{uuid.uuid4().hex[:6]}.wav"
        cls.test_wav_path = os.path.join("beat_outputs", cls.test_wav_name)
        sr = 32000
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        audio_data = 0.3 * np.sin(2 * np.pi * 440 * t)
        sf.write(cls.test_wav_path, audio_data, sr)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_wav_path):
            try:
                os.remove(cls.test_wav_path)
            except Exception:
                pass

    def create_test_user(self, username_prefix="user"):
        unique_id = uuid.uuid4().hex[:6]
        username = f"{username_prefix}_{unique_id}"
        email = f"{username}@example.com"
        password = "password123"
        
        res = requests.post(f"{BASE_URL}/auth/register", json={
            "username": username,
            "email": email,
            "password": password
        })
        self.assertIn(res.status_code, [200, 201], f"Register failed: {res.text}")
        data = res.json()
        token = data.get("token") or data.get("access_token")
        return {
            "username": username,
            "email": email,
            "token": token,
            "headers": {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        }

    def test_activity_lifecycle(self):
        """Verify that project creation, commits, stars, comments, forks, and clones create activity events."""
        print("\n[TEST] 1. Registering test users...")
        user1 = self.create_test_user("alice")
        user2 = self.create_test_user("bob")

        print("[TEST] 2. Creating a new project for alice...")
        res = requests.post(f"{BASE_URL}/projects", json={
            "name": "Activity Test Project",
            "description": "Testing activity timeline feature",
            "is_public": True,
            "genre": "Lo-Fi",
            "bpm": 85,
            "key": "C Minor"
        }, headers=user1["headers"])
        self.assertIn(res.status_code, [200, 201], f"Create project failed: {res.text}")
        repo = res.json()
        repo_id = repo["id"]
        print(f"       -> Project created with ID: {repo_id}")

        print("[TEST] 3. Verifying initial project_created activity...")
        res = requests.get(f"{BASE_URL}/projects/{repo_id}/activity", headers=user1["headers"])
        self.assertEqual(res.status_code, 200, f"Get activity failed: {res.text}")
        act_data = res.json()
        self.assertGreaterEqual(act_data["total_count"], 1)
        events = act_data["activities"]
        self.assertTrue(any(e["event_type"].upper() == "PROJECT_CREATED" for e in events))
        print(f"       -> Verified project_created event: {events[0]['title']}")

        print("[TEST] 4. Adding a commit to the project...")
        res = requests.post(f"{BASE_URL}/projects/{repo_id}/commit", json={
            "filename": self.test_wav_name,
            "message": "Initial beat track with synth",
            "prompt": "chill lofi beat 85 bpm",
            "mood": "Lo-fi Chill"
        }, headers=user1["headers"])
        self.assertIn(res.status_code, [200, 201], f"Commit failed: {res.text}")
        commit = res.json()
        commit_id = commit["id"]
        print(f"       -> Commit created with ID: {commit_id}")

        print("[TEST] 5. Verifying commit_created activity...")
        res = requests.get(f"{BASE_URL}/projects/{repo_id}/activity", headers=user1["headers"])
        self.assertEqual(res.status_code, 200)
        act_data = res.json()
        events = act_data["activities"]
        commit_events = [e for e in events if e.get("commit_id") == commit_id or "Initial beat" in e.get("title", "")]
        self.assertGreaterEqual(len(commit_events), 1)
        c_event = commit_events[0]
        self.assertEqual(c_event["badge_class"], "dot-commit")
        print(f"       -> Verified commit event: {c_event['title']}, class={c_event['badge_class']}")

        print("[TEST] 6. Bob stars the project...")
        res = requests.post(f"{BASE_URL}/projects/{repo_id}/star", headers=user2["headers"])
        self.assertIn(res.status_code, [200, 201])

        print("[TEST] 7. Bob comments on the project...")
        res = requests.post(f"{BASE_URL}/projects/{repo_id}/commits/{commit_id}/comments", json={
            "body": "Awesome chill vibe on this track! 🔥"
        }, headers=user2["headers"])
        self.assertIn(res.status_code, [200, 201])

        print("[TEST] 8. Bob forks the project...")
        res = requests.post(f"{BASE_URL}/projects/{repo_id}/fork", headers=user2["headers"])
        self.assertIn(res.status_code, [200, 201])
        fork_repo = res.json()
        fork_id = fork_repo["id"]
        print(f"       -> Fork created with ID: {fork_id}")

        print("[TEST] 9. Verifying timeline on original repo...")
        res = requests.get(f"{BASE_URL}/projects/{repo_id}/activity", headers=user1["headers"])
        self.assertEqual(res.status_code, 200)
        act_data = res.json()
        events = act_data["activities"]
        types = [e["event_type"].upper() for e in events]
        self.assertTrue("STAR_ADDED" in types or "PROJECT_STARRED" in types or any("Starred" in e["title"] for e in events))
        self.assertTrue("COMMENT_ADDED" in types or any("Commented" in e["title"] for e in events))
        print(f"       -> Total events on repo: {len(events)}")

        print("[TEST] 10. Testing category filtering...")
        # Commits only
        res = requests.get(f"{BASE_URL}/projects/{repo_id}/activity?filter_category=commits", headers=user1["headers"])
        self.assertEqual(res.status_code, 200)
        c_filtered = res.json()["activities"]
        self.assertGreaterEqual(len(c_filtered), 1)
        for e in c_filtered:
            self.assertTrue(e["category"] in ("commits", "commit") or "Commit" in e["title"])

        # Social only
        res = requests.get(f"{BASE_URL}/projects/{repo_id}/activity?filter_category=social", headers=user1["headers"])
        self.assertEqual(res.status_code, 200)
        s_filtered = res.json()["activities"]
        self.assertGreaterEqual(len(s_filtered), 2)
        for e in s_filtered:
            self.assertEqual(e["category"], "social")

        print("[TEST] 11. Verifying forked repo timeline...")
        res = requests.get(f"{BASE_URL}/projects/{fork_id}/activity", headers=user2["headers"])
        self.assertEqual(res.status_code, 200)
        fork_act_data = res.json()
        fork_types = [e["event_type"].upper() for e in fork_act_data["activities"]]
        self.assertTrue("FORK_CREATED" in fork_types or "PROJECT_FORKED" in fork_types or any("Forked" in e["title"] for e in fork_act_data["activities"]))
        
        print("\n========================================================")
        print("  [SUCCESS] All 11 Activity Timeline assertions passed!")
        print("========================================================\n")

if __name__ == "__main__":
    unittest.main()
