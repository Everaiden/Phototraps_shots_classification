import React from "react";
import { Container } from "react-bootstrap";
import Header from "./components/Header";
import Slider from "./components/Slider";
import Footer from "./components/Footer";

const App = () => {
  return (
    <>
      <Header />
      <main>
        <Container>
          <Slider />
        </Container>
      </main>
      <Footer />
    </>
  );
}

export default App;
