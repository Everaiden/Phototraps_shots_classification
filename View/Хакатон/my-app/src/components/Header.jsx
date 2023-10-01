import React from 'react'
import { Container } from 'react-bootstrap'

const Header = () => {
    return <header>
        <Container>
            <nav class="navbar navbar-expand-lg bg-body-tertiary">
                <div class="container-fluid">
                    <a class="navbar-brand" href="#">SOS</a>
                    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                    </button>
                    <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav">
                        <li class="nav-item">
                        <a class="nav-link active" aria-current="page" href="#">Главная</a>
                        </li>
                        <li class="nav-item">
                        <a class="nav-link" href="#">О программе</a>
                        </li>
                        <li class="nav-item">
                        <a class="nav-link" href="#">Справка</a>
                        </li>
                    </ul>
                    </div>
                </div>
            </nav>
        </Container>
    </header>
}

export default Header