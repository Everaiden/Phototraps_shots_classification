import React from "react";
import { Row, Col, Container } from "react-bootstrap";
import Header from "./components/Header";
import Slider from "./components/Slider";
import Footer from "./components/Footer";
import FormFiles from "./components/FormFiles";

const App = () => {
  return (
    <>
      <Header />
      <main>
        <Container>
          <h1>Классификация снимков с фотоловушек</h1>
          <Row>
            <Col className="col-8">
              <Slider />
            </Col>
            <Col className="col-4">
              <FormFiles/>
            </Col>
            
          </Row>
        </Container>
      </main>
      <Footer />
    </>
  );
}

export default App;
