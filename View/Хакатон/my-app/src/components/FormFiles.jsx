import React from 'react'
import {Form} from 'react-bootstrap'

const FormFiles = () => {
  return (
    <Form.Group controlId="formFileMultiple">
      <Form.Label>Вставьте изображение</Form.Label>
      <Form.Control type="file"/>
    </Form.Group>
)}

export default FormFiles