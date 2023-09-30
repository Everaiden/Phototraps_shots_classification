import React from "react"
import { Carousel, CarouselCaption, CarouselItem } from "react-bootstrap"
import NeverGonna1 from "./Images/Never-Gonna1.jpg"
import NeverGonna2 from "./Images/Never-Gonna2.jpg"
import NeverGonna3 from "./Images/Never-Gonna3.jpg"

export default function Slider (){
    return (
    <Carousel>
        <Carousel.Item>
            <img 
                className="d-block w-100"
                src={NeverGonna1}
                alt="Never gonna give you up"
            />
            <Carousel.Caption>
                <h3>Never gonna give you up</h3>
            </Carousel.Caption>
        </Carousel.Item>
        <Carousel.Item>
            <img 
                className="d-block w-100"
                src={NeverGonna2}
                alt="Never gonna let you down"
            />
            <Carousel.Caption>
                <h3>Never gonna let you down</h3>
            </Carousel.Caption>
        </Carousel.Item>
        <Carousel.Item>
            <img 
                className="d-block w-100"
                src={NeverGonna3}
                alt="Never gonna run around and desert you"
            />
            <Carousel.Caption>
                <h3>Never gonna run around and desert you</h3>
            </Carousel.Caption>
        </Carousel.Item>
    </Carousel>
)}

