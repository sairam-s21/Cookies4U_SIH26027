import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import CorridorMapPage from "./pages/CorridorMapPage.jsx";
import WaitingList from "./pages/WaitingList.jsx";
import RecommendedScheduling from "./pages/RecommendedScheduling.jsx";
import Schedule from "./pages/Schedule.jsx";
import ApprovedTasks from "./pages/ApprovedTasks.jsx";
import CompletedHistory from "./pages/CompletedHistory.jsx";
import EmergencyHandling from "./pages/EmergencyHandling.jsx";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="corridor-map" element={<CorridorMapPage />} />
          <Route path="waiting-list" element={<WaitingList />} />
          <Route path="recommended-scheduling" element={<RecommendedScheduling />} />
          <Route path="schedule" element={<Schedule />} />
          <Route path="approved-tasks" element={<ApprovedTasks />} />
          <Route path="completed-history" element={<CompletedHistory />} />
          <Route path="emergency-handling" element={<EmergencyHandling />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
