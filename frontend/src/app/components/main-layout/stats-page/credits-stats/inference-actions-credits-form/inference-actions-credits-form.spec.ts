import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InferenceActionsCreditsForm } from './inference-actions-credits-form';

describe('InferenceActionsCreditsForm', () => {
  let component: InferenceActionsCreditsForm;
  let fixture: ComponentFixture<InferenceActionsCreditsForm>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InferenceActionsCreditsForm],
    }).compileComponents();

    fixture = TestBed.createComponent(InferenceActionsCreditsForm);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
