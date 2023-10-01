import React from 'react'
import {Button, Form} from 'react-bootstrap'

const FormFiles = () => {
  return (
    <div id='FormFiles'>
      <Form.Group controlId="formFileMultiple">
        <Form.Label />
        <Form.Control type="file"/>
        <div className="d-grid gap-2">
          <Button variant="primary" size="lg">
            Запуск
          </Button>
        </div>
      </Form.Group>
    </div>
)}

export default FormFiles