import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './components/HomePage';
import CapturePage from './components/CapturePage';
import ResultPage from './components/ResultPage';
import Layout from './components/Layout';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/capture" element={<CapturePage />} />
          <Route path="/results" element={<ResultPage />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;