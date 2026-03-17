import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FormSubmitDetail } from './form-submit-detail';

describe('FormSubmitDetail', () => {
  let component: FormSubmitDetail;
  let fixture: ComponentFixture<FormSubmitDetail>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FormSubmitDetail],
    }).compileComponents();

    fixture = TestBed.createComponent(FormSubmitDetail);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
