import Card from "../components/Card.jsx";
import EmergencyPanel from "../components/EmergencyPanel.jsx";

export default function EmergencyHandling() {
  return (
    <div className="stack">
      <Card title="Emergency handling">
        <EmergencyPanel />
      </Card>
    </div>
  );
}
