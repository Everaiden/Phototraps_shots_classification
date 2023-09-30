import React from "react"
import { Carousel } from "react-bootstrap"
import Photo1 from "./Images/Photo1.jpeg"
import Photo2 from "./Images/Photo2.jpeg"
import Photo3 from "./Images/Photo3.jpeg"

export default function Slider (){
    return (
        <Carousel>
            <Carousel.Item>
                <img 
                    className="d-block w-100"
                    src={Photo1}
                     alt=". . ."
                />
                <Carousel.Caption>
                    <h3>Енотики</h3>
                </Carousel.Caption>
            </Carousel.Item>
            <Carousel.Item>
                <img 
                    className="d-block w-100"
                    src={Photo2}
                    alt=". . . "
                />
                <Carousel.Caption>
                    <h3>Кролик</h3>
                </Carousel.Caption>
            </Carousel.Item>
            <Carousel.Item>
                <img 
                    className="d-block w-100"
                    src={Photo3}
                    alt=". . . "
                />
                <Carousel.Caption>
                    <h3>Красная Панда</h3>
                </Carousel.Caption>
            </Carousel.Item>
        </Carousel>
)}

