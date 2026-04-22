import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ConfirmationPopup } from './confirmation-popup';

describe('ConfirmationPopup', () => {
  let component: ConfirmationPopup;
  let fixture: ComponentFixture<ConfirmationPopup>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ConfirmationPopup],
    }).compileComponents();

    fixture = TestBed.createComponent(ConfirmationPopup);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
