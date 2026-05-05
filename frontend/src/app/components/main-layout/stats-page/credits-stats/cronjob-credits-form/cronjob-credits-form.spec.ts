import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CronjobCreditsForm } from './cronjob-credits-form';

describe('CronjobCreditsForm', () => {
  let component: CronjobCreditsForm;
  let fixture: ComponentFixture<CronjobCreditsForm>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CronjobCreditsForm],
    }).compileComponents();

    fixture = TestBed.createComponent(CronjobCreditsForm);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
